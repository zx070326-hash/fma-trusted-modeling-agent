from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from random import Random
from typing import Annotated, Literal
from uuid import uuid4

import numpy as np
from pydantic import Field, model_validator
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
from scipy.signal import savgol_filter
from scipy.stats import beta

from fma.hashing import sha256_value
from fma.schemas import ArtifactRef, StrictModel
from fma.storage import RunStore
from fma.v2.dynamics_ir import (
    PolynomialBasisTermV24,
    evaluate_polynomial_library,
    polynomial_basis_terms,
    trajectory_nrmse,
)
from fma.v2.schemas import Identifier, Sha256, _assert_timezone

from .model_challenge_v37 import _hash_without
from .representation_invariant_topology_v311 import (
    RepresentationTopologyManifestV311,
    RepresentationTopologyReportV311,
    verify_representation_topology_run_v311,
)
from . import representation_invariant_topology_v311 as v311_module


DEVELOPMENT_SEEDS_V312 = (28019, 28081, 28151, 28219, 28283, 28349)
CONFIRMATION_SEEDS_V312 = (
    29017, 29077, 29147, 29221, 29287,
    29363, 29429, 29501, 29573, 29641,
)

MechanismV312 = Literal["gompertz_open_set", "pendulum_open_set"]
RepresentationV312 = Literal["anonymous_reference", "anonymous_scaled_permuted"]
ConceptV312 = Literal[
    "generic_degree_1",
    "generic_degree_2",
    "generic_degree_3",
    "generic_degree_4",
    "logarithmic_rate",
    "saturating_rate_decoy",
    "scalar_affine_decoy",
    "periodic_restoring_force",
    "kinematic_cubic_decoy",
    "uncoupled_linear_decoy",
]

MECHANISMS_V312: tuple[MechanismV312, ...] = (
    "gompertz_open_set",
    "pendulum_open_set",
)
REPRESENTATIONS_V312: tuple[RepresentationV312, ...] = (
    "anonymous_reference",
    "anonymous_scaled_permuted",
)
BASELINE_CONCEPTS_V312: tuple[ConceptV312, ...] = (
    "generic_degree_1",
    "generic_degree_2",
    "generic_degree_3",
    "generic_degree_4",
)
CANDIDATE_CONCEPTS_BY_DIMENSION_V312: dict[int, tuple[ConceptV312, ...]] = {
    1: (
        "generic_degree_3",
        "logarithmic_rate",
        "saturating_rate_decoy",
        "scalar_affine_decoy",
    ),
    2: (
        "generic_degree_3",
        "periodic_restoring_force",
        "kinematic_cubic_decoy",
        "uncoupled_linear_decoy",
    ),
}
EXPECTED_CONCEPT_V312: dict[MechanismV312, ConceptV312] = {
    "gompertz_open_set": "logarithmic_rate",
    "pendulum_open_set": "periodic_restoring_force",
}


def _file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _committed_refs(store: RunStore) -> list[ArtifactRef]:
    events = [
        json.loads(line)
        for line in store.event_path.read_text(encoding="utf-8").splitlines()
    ]
    return [
        ArtifactRef.model_validate(event["payload"])
        for event in events if event["event_type"] == "artifact_committed"
    ]


def _load_one(store: RunStore, refs: list[ArtifactRef], kind: str, model):
    matches = [item for item in refs if item.kind == kind]
    if len(matches) != 1:
        raise ValueError(f"V3.12 expected one {kind}")
    return model.model_validate(store.load_artifact(matches[0]))


class VerifiedLineageReceiptV312(StrictModel):
    schema_version: Literal["3.12"] = "3.12"
    receipt_id: Identifier
    source_run_id: Identifier
    source_report_hash: Sha256
    source_manifest_hash: Sha256
    source_event_head_hash: Sha256
    source_artifact_refs: list[ArtifactRef] = Field(min_length=9, max_length=9)
    source_verifier_file_hash: Sha256
    full_replay_verified: Literal[True] = True
    verified_at: datetime
    receipt_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_receipt(self) -> "VerifiedLineageReceiptV312":
        _assert_timezone(self.verified_at, "verified_at")
        if self.receipt_hash and self.receipt_hash != self.content_hash():
            raise ValueError("V3.12 lineage receipt hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "receipt_hash")

    def assert_sealed(self) -> None:
        if not self.receipt_hash or self.receipt_hash != self.content_hash():
            raise ValueError("V3.12 lineage receipt is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "VerifiedLineageReceiptV312":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"receipt_hash"}),
            receipt_hash=draft.content_hash(),
        )


def build_verified_v311_lineage_receipt_v312(
    source_run_directory: str | Path,
    *,
    source_development_run_directory: str | Path,
    verified_at: datetime | None = None,
) -> VerifiedLineageReceiptV312:
    if not verify_representation_topology_run_v311(
        source_run_directory,
        source_v310_run_directory=(
            Path(source_run_directory).resolve().parents[1]
            / "iteration_18" / "v310_skeleton_factorial"
        ),
        development_run_directory=source_development_run_directory,
    ):
        raise ValueError("V3.12 source V3.11 did not fully replay")
    store = RunStore.open_existing(source_run_directory)
    refs = _committed_refs(store)
    report = _load_one(
        store, refs, "representation_topology_report_v311",
        RepresentationTopologyReportV311,
    )
    manifest = _load_one(
        store, refs, "representation_topology_manifest_v311",
        RepresentationTopologyManifestV311,
    )
    if report.status != "representation_topology_confirmed_v311":
        raise ValueError("V3.12 lineage source is not confirmed")
    last_event = json.loads(
        store.event_path.read_text(encoding="utf-8").splitlines()[-1]
    )
    return VerifiedLineageReceiptV312.seal(
        receipt_id="verified_v311_lineage_for_v312",
        source_run_id=store.run_id,
        source_report_hash=report.report_hash,
        source_manifest_hash=manifest.manifest_hash,
        source_event_head_hash=last_event["event_hash"],
        source_artifact_refs=refs,
        source_verifier_file_hash=_file_sha256(v311_module.__file__),
        verified_at=verified_at or datetime.now(timezone.utc),
    )


def verify_v311_lineage_receipt_v312(
    receipt: VerifiedLineageReceiptV312,
    source_run_directory: str | Path,
) -> bool:
    try:
        receipt.assert_sealed()
        if receipt.verified_at > datetime.now(timezone.utc) + timedelta(minutes=5):
            return False
        if _file_sha256(v311_module.__file__) != receipt.source_verifier_file_hash:
            return False
        store = RunStore.open_existing(source_run_directory)
        if store.run_id != receipt.source_run_id or not store.verify_event_chain():
            return False
        events = [
            json.loads(line)
            for line in store.event_path.read_text(encoding="utf-8").splitlines()
        ]
        if events[-1]["event_hash"] != receipt.source_event_head_hash:
            return False
        refs = _committed_refs(store)
        if [item.model_dump(mode="json") for item in refs] != [
            item.model_dump(mode="json") for item in receipt.source_artifact_refs
        ]:
            return False
        for ref in refs:
            store.load_artifact(ref)
        report = _load_one(
            store, refs, "representation_topology_report_v311",
            RepresentationTopologyReportV311,
        )
        manifest = _load_one(
            store, refs, "representation_topology_manifest_v311",
            RepresentationTopologyManifestV311,
        )
        return (
            report.status == "representation_topology_confirmed_v311"
            and report.report_hash == receipt.source_report_hash
            and manifest.manifest_hash == receipt.source_manifest_hash
        )
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


class ConceptMethodSourceV312(StrictModel):
    source_id: Identifier
    title: str = Field(min_length=5)
    source_url: str = Field(min_length=10)
    borrowed_principle: str = Field(min_length=10)
    non_transfer_limit: str = Field(min_length=10)


class ConceptMethodEvidenceV312(StrictModel):
    schema_version: Literal["3.12"] = "3.12"
    evidence_id: Identifier
    retrieval_date: Literal["2026-07-22"] = "2026-07-22"
    retrieval_scope: Literal["targeted_primary_source_not_systematic_review"] = (
        "targeted_primary_source_not_systematic_review"
    )
    sources: list[ConceptMethodSourceV312] = Field(min_length=5, max_length=5)
    evidence_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_evidence(self) -> "ConceptMethodEvidenceV312":
        if len({item.source_id for item in self.sources}) != len(self.sources):
            raise ValueError("V3.12 method sources differ")
        if self.evidence_hash and self.evidence_hash != self.content_hash():
            raise ValueError("V3.12 method evidence hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "evidence_hash")

    def assert_sealed(self) -> None:
        if not self.evidence_hash or self.evidence_hash != self.content_hash():
            raise ValueError("V3.12 method evidence is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ConceptMethodEvidenceV312":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"evidence_hash"}),
            evidence_hash=draft.content_hash(),
        )


def default_concept_method_evidence_v312() -> ConceptMethodEvidenceV312:
    return ConceptMethodEvidenceV312.seal(
        evidence_id="open_set_concept_evolution_method_evidence_v312",
        sources=[
            ConceptMethodSourceV312(
                source_id="ai_feynman_2_2020",
                title="AI Feynman 2.0 Pareto-optimal symbolic regression exploiting graph modularity",
                source_url=(
                    "https://proceedings.neurips.cc/paper_files/paper/2020/hash/"
                    "33a854e247155d590883b93bca53848a-Abstract.html"
                ),
                borrowed_principle="Challenge expression accuracy and complexity jointly on a Pareto frontier.",
                non_transfer_limit="Feynman benchmark performance is not evidence for noisy dynamical OOD recovery.",
            ),
            ConceptMethodSourceV312(
                source_id="lasr_2024",
                title="Symbolic Regression with a Learned Concept Library",
                source_url=(
                    "https://proceedings.neurips.cc/paper_files/paper/2024/hash/"
                    "4ec3ddc465c6d650c9c419fb91f1c00a-Abstract-Conference.html"
                ),
                borrowed_principle="Separate reusable concept abstraction from ordinary expression evolution.",
                non_transfer_limit="Reported benchmark lift does not approve a concept for a new mechanism.",
            ),
            ConceptMethodSourceV312(
                source_id="drsr_2025",
                title="DrSR LLM based Scientific Equation Discovery with Dual Reasoning",
                source_url="https://arxiv.org/abs/2506.04282",
                borrowed_principle="Keep data-aware signatures separate from experience-derived proposal ideas.",
                non_transfer_limit="Preprint cross-domain results require independent local reproduction.",
            ),
            ConceptMethodSourceV312(
                source_id="sr_scientist_2025",
                title="SR-Scientist Scientific Equation Discovery With Agentic AI",
                source_url="https://arxiv.org/abs/2510.11661",
                borrowed_principle="Use bounded data-analysis and equation-evaluation tools over a persistent search horizon.",
                non_transfer_limit="Observed-score selection and trimmed errors cannot replace private adjudication.",
            ),
            ConceptMethodSourceV312(
                source_id="restart_2026",
                title="Robust Equation Structure Learning with Adaptive Refinement",
                source_url="https://openreview.net/forum?id=z9TKJhLVKj",
                borrowed_principle="Turn unexplained residual structure into short-term refinement and long-term concepts.",
                non_transfer_limit="Paper acceptance and benchmark claims do not establish FMA concept admission.",
            ),
        ],
    )


class ConceptEvolutionPolicyV312(StrictModel):
    schema_version: Literal["3.12"] = "3.12"
    policy_id: Identifier
    method_evidence_hash: Sha256
    lineage_receipt_hash: Sha256
    baseline_concepts: list[ConceptV312] = Field(min_length=4, max_length=4)
    candidate_concepts_by_dimension: dict[int, list[ConceptV312]]
    expression_evaluations_per_arm: Literal[4] = 4
    residual_signature_used_for_proposal_order_only: Literal[True] = True
    arbitrary_code_execution_permitted: Literal[False] = False
    private_mechanism_visible: Literal[False] = False
    private_representation_visible: Literal[False] = False
    private_probe_visible: Literal[False] = False
    private_loss_visible: Literal[False] = False
    public_score_can_admit_concept: Literal[False] = False
    persist_all_attempts: Literal[True] = True
    task_router_permitted: Literal[False] = False
    real_world_execution_permitted: Literal[False] = False
    policy_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_policy(self) -> "ConceptEvolutionPolicyV312":
        if self.baseline_concepts != list(BASELINE_CONCEPTS_V312):
            raise ValueError("V3.12 baseline concept budget differs")
        expected = {
            key: list(value)
            for key, value in CANDIDATE_CONCEPTS_BY_DIMENSION_V312.items()
        }
        if self.candidate_concepts_by_dimension != expected:
            raise ValueError("V3.12 candidate grammar differs")
        if self.policy_hash and self.policy_hash != self.content_hash():
            raise ValueError("V3.12 policy hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "policy_hash")

    def assert_sealed(self) -> None:
        if not self.policy_hash or self.policy_hash != self.content_hash():
            raise ValueError("V3.12 policy is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ConceptEvolutionPolicyV312":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"policy_hash"}),
            policy_hash=draft.content_hash(),
        )


def default_concept_evolution_policy_v312(
    evidence: ConceptMethodEvidenceV312,
    lineage: VerifiedLineageReceiptV312,
) -> ConceptEvolutionPolicyV312:
    evidence.assert_sealed()
    lineage.assert_sealed()
    return ConceptEvolutionPolicyV312.seal(
        policy_id="bounded_residual_guided_concept_evolution_v312",
        method_evidence_hash=evidence.evidence_hash,
        lineage_receipt_hash=lineage.receipt_hash,
        baseline_concepts=list(BASELINE_CONCEPTS_V312),
        candidate_concepts_by_dimension={
            key: list(value)
            for key, value in CANDIDATE_CONCEPTS_BY_DIMENSION_V312.items()
        },
    )


class PublicConceptProtocolV312(StrictModel):
    schema_version: Literal["3.12"] = "3.12"
    protocol_id: Identifier
    trajectory_points: Literal[81] = 81
    time_step: Literal[0.04] = 0.04
    public_trajectory_count: Literal[4] = 4
    fit_trajectory_count: Literal[2] = 2
    validation_trajectory_index: Literal[2] = 2
    challenge_trajectory_index: Literal[3] = 3
    savgol_window: Literal[11] = 11
    savgol_polynomial: Literal[3] = 3
    ridge_alpha: Literal[1e-8] = 1e-8
    complexity_penalty: Literal[0.001] = 0.001
    maximum_public_score: Literal[0.5] = 0.5
    unresolved_loss: Literal[10.0] = 10.0
    method_evidence_hash: Sha256
    policy_hash: Sha256
    lineage_receipt_hash: Sha256
    frozen_at: datetime
    protocol_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_protocol(self) -> "PublicConceptProtocolV312":
        _assert_timezone(self.frozen_at, "frozen_at")
        if self.protocol_hash and self.protocol_hash != self.content_hash():
            raise ValueError("V3.12 public protocol hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "protocol_hash")

    def assert_sealed(self) -> None:
        if not self.protocol_hash or self.protocol_hash != self.content_hash():
            raise ValueError("V3.12 public protocol is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "PublicConceptProtocolV312":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"protocol_hash"}),
            protocol_hash=draft.content_hash(),
        )


def default_public_concept_protocol_v312(
    evidence: ConceptMethodEvidenceV312,
    lineage: VerifiedLineageReceiptV312,
    policy: ConceptEvolutionPolicyV312,
    *,
    frozen_at: datetime | None = None,
) -> PublicConceptProtocolV312:
    for item in (evidence, lineage, policy):
        item.assert_sealed()
    return PublicConceptProtocolV312.seal(
        protocol_id="open_set_concept_public_protocol_v312",
        method_evidence_hash=evidence.evidence_hash,
        policy_hash=policy.policy_hash,
        lineage_receipt_hash=lineage.receipt_hash,
        frozen_at=frozen_at or datetime.now(timezone.utc),
    )


class PrivateConceptWorldPackSpecV312(StrictModel):
    schema_version: Literal["3.12"] = "3.12"
    experiment_id: Identifier
    phase: Literal["development", "confirmation"]
    mechanisms: list[MechanismV312] = Field(min_length=2, max_length=2)
    representations: list[RepresentationV312] = Field(min_length=2, max_length=2)
    seeds: list[int] = Field(min_length=6, max_length=10)
    observation_noise_fraction: Literal[0.0015] = 0.0015
    calibration_failure_seed_index: Literal[0] = 0
    expected_quality_case_count: Literal[4] = 4
    public_protocol_hash: Sha256
    lineage_receipt_hash: Sha256
    development_report_hash: Sha256 | None = None
    bootstrap_replicates: Literal[2000] = 2000
    bootstrap_seed: int
    material_negative_transfer: Literal[0.02] = 0.02
    maximum_negative_transfer_upper_95: Literal[0.1] = 0.1
    minimum_concept_recovery_accuracy: Literal[0.9] = 0.9
    minimum_pair_concept_consistency: Literal[0.9] = 0.9
    maximum_pair_loss_difference: Literal[0.05] = 0.05
    maximum_mechanism_regression: Literal[0.02] = 0.02
    maximum_transformed_representation_regression: Literal[0.02] = 0.02
    frozen_delta: Literal[
        "two_open_set_operator_families_residual_guided_concept_evolution_only"
    ] = "two_open_set_operator_families_residual_guided_concept_evolution_only"
    frozen_at: datetime
    spec_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_spec(self) -> "PrivateConceptWorldPackSpecV312":
        _assert_timezone(self.frozen_at, "frozen_at")
        if self.mechanisms != list(MECHANISMS_V312):
            raise ValueError("V3.12 mechanism order differs")
        if self.representations != list(REPRESENTATIONS_V312):
            raise ValueError("V3.12 representation order differs")
        expected = (
            list(DEVELOPMENT_SEEDS_V312)
            if self.phase == "development" else list(CONFIRMATION_SEEDS_V312)
        )
        if self.seeds != expected:
            raise ValueError("V3.12 seeds differ from frozen phase")
        if self.phase == "development" and self.development_report_hash is not None:
            raise ValueError("V3.12 development spec cannot bind itself")
        if self.phase == "confirmation" and self.development_report_hash is None:
            raise ValueError("V3.12 confirmation spec needs development lineage")
        if self.spec_hash and self.spec_hash != self.content_hash():
            raise ValueError("V3.12 private spec hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "spec_hash")

    def assert_sealed(self) -> None:
        if not self.spec_hash or self.spec_hash != self.content_hash():
            raise ValueError("V3.12 private spec is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "PrivateConceptWorldPackSpecV312":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"spec_hash"}),
            spec_hash=draft.content_hash(),
        )


def default_private_concept_spec_v312(
    protocol: PublicConceptProtocolV312,
    lineage: VerifiedLineageReceiptV312,
    *,
    phase: Literal["development", "confirmation"],
    development_report_hash: str | None = None,
    frozen_at: datetime | None = None,
) -> PrivateConceptWorldPackSpecV312:
    protocol.assert_sealed()
    lineage.assert_sealed()
    return PrivateConceptWorldPackSpecV312.seal(
        experiment_id=f"open_set_concept_evolution_{phase}_v312",
        phase=phase,
        mechanisms=list(MECHANISMS_V312),
        representations=list(REPRESENTATIONS_V312),
        seeds=(
            list(DEVELOPMENT_SEEDS_V312)
            if phase == "development" else list(CONFIRMATION_SEEDS_V312)
        ),
        public_protocol_hash=protocol.protocol_hash,
        lineage_receipt_hash=lineage.receipt_hash,
        development_report_hash=development_report_hash,
        bootstrap_seed=(3120722 if phase == "development" else 3121722),
        frozen_at=frozen_at or datetime.now(timezone.utc),
    )


class AnonymousConceptTrajectoryV312(StrictModel):
    trajectory_id: Identifier
    case_id: Identifier
    times: list[Annotated[float, Field(allow_inf_nan=False)]] = Field(min_length=81, max_length=81)
    states: list[list[Annotated[float, Field(allow_inf_nan=False)]]] = Field(min_length=81, max_length=81)
    trajectory_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_trajectory(self) -> "AnonymousConceptTrajectoryV312":
        if len(self.times) != len(self.states):
            raise ValueError("V3.12 trajectory arrays differ")
        if any(right <= left for left, right in zip(self.times, self.times[1:])):
            raise ValueError("V3.12 times must increase")
        if self.trajectory_hash and self.trajectory_hash != self.content_hash():
            raise ValueError("V3.12 trajectory hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "trajectory_hash")

    def assert_sealed(self) -> None:
        if not self.trajectory_hash or self.trajectory_hash != self.content_hash():
            raise ValueError("V3.12 trajectory is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "AnonymousConceptTrajectoryV312":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"trajectory_hash"}),
            trajectory_hash=draft.content_hash(),
        )


class PublicConceptCaseV312(StrictModel):
    schema_version: Literal["3.12"] = "3.12"
    case_id: Identifier
    state_names: list[Identifier] = Field(min_length=1, max_length=2)
    trajectories: list[AnonymousConceptTrajectoryV312] = Field(min_length=4, max_length=4)
    quality_flags: list[Identifier]
    semantic_state_labels_available: Literal[False] = False
    representation_metadata_available: Literal[False] = False
    public_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_case(self) -> "PublicConceptCaseV312":
        if self.state_names != [f"z{index}" for index in range(len(self.state_names))]:
            raise ValueError("V3.12 state names must be anonymous")
        for trajectory in self.trajectories:
            trajectory.assert_sealed()
            if trajectory.case_id != self.case_id:
                raise ValueError("V3.12 trajectory case differs")
            if any(len(row) != len(self.state_names) for row in trajectory.states):
                raise ValueError("V3.12 trajectory dimension differs")
        if self.public_hash and self.public_hash != self.content_hash():
            raise ValueError("V3.12 public case hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "public_hash")

    def assert_sealed(self) -> None:
        if not self.public_hash or self.public_hash != self.content_hash():
            raise ValueError("V3.12 public case is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "PublicConceptCaseV312":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"public_hash"}),
            public_hash=draft.content_hash(),
        )


class PublicConceptWorldPackV312(StrictModel):
    schema_version: Literal["3.12"] = "3.12"
    public_protocol_hash: Sha256
    cases: list[PublicConceptCaseV312] = Field(min_length=24, max_length=40)
    public_pack_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_pack(self) -> "PublicConceptWorldPackV312":
        ids = [item.case_id for item in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("V3.12 public case ids differ")
        for case in self.cases:
            case.assert_sealed()
        if self.public_pack_hash and self.public_pack_hash != self.content_hash():
            raise ValueError("V3.12 public pack hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "public_pack_hash")

    def assert_sealed(self) -> None:
        if not self.public_pack_hash or self.public_pack_hash != self.content_hash():
            raise ValueError("V3.12 public pack is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "PublicConceptWorldPackV312":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"public_pack_hash"}),
            public_pack_hash=draft.content_hash(),
        )


class PrivateConceptCaseV312(StrictModel):
    schema_version: Literal["3.12"] = "3.12"
    public_case: PublicConceptCaseV312
    mechanism: MechanismV312
    representation: RepresentationV312
    expected_concept: ConceptV312
    hidden_pair_id: Identifier
    hidden_seed: int
    hidden_parameters: dict[str, Annotated[float, Field(allow_inf_nan=False)]]
    observed_to_physical_permutation: list[Annotated[int, Field(ge=0, le=1)]]
    observed_scales: list[Annotated[float, Field(gt=0, allow_inf_nan=False)]]
    private_probe_initials: list[list[Annotated[float, Field(allow_inf_nan=False)]]] = Field(min_length=3, max_length=3)
    private_probe_truths: list[list[list[Annotated[float, Field(allow_inf_nan=False)]]]] = Field(min_length=3, max_length=3)
    private_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_private(self) -> "PrivateConceptCaseV312":
        self.public_case.assert_sealed()
        dimension = len(self.public_case.state_names)
        if sorted(self.observed_to_physical_permutation) != list(range(dimension)):
            raise ValueError("V3.12 hidden permutation differs")
        if len(self.observed_scales) != dimension:
            raise ValueError("V3.12 hidden scales differ")
        if self.expected_concept != EXPECTED_CONCEPT_V312[self.mechanism]:
            raise ValueError("V3.12 expected concept differs")
        if self.private_hash and self.private_hash != self.content_hash():
            raise ValueError("V3.12 private case hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "private_hash")

    def assert_sealed(self) -> None:
        if not self.private_hash or self.private_hash != self.content_hash():
            raise ValueError("V3.12 private case is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "PrivateConceptCaseV312":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"private_hash"}),
            private_hash=draft.content_hash(),
        )


class PrivateConceptWorldPackV312(StrictModel):
    schema_version: Literal["3.12"] = "3.12"
    spec_hash: Sha256
    public_pack_hash: Sha256
    cases: list[PrivateConceptCaseV312] = Field(min_length=24, max_length=40)
    generated_at: datetime
    pack_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_pack(self) -> "PrivateConceptWorldPackV312":
        _assert_timezone(self.generated_at, "generated_at")
        ids = [item.public_case.case_id for item in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("V3.12 private case ids differ")
        for case in self.cases:
            case.assert_sealed()
        if self.pack_hash and self.pack_hash != self.content_hash():
            raise ValueError("V3.12 private pack hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "pack_hash")

    def assert_sealed(self) -> None:
        if not self.pack_hash or self.pack_hash != self.content_hash():
            raise ValueError("V3.12 private pack is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "PrivateConceptWorldPackV312":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"pack_hash"}),
            pack_hash=draft.content_hash(),
        )


def _mechanism_setup_v312(
    mechanism: MechanismV312,
    random: Random,
) -> tuple[dict[str, float], list[list[float]], list[list[float]]]:
    if mechanism == "gompertz_open_set":
        parameters = {
            "rate": 0.48 + 0.25 * random.random(),
            "capacity": 2.6 + 1.4 * random.random(),
        }
        capacity = parameters["capacity"]
        return (
            parameters,
            [[factor * capacity] for factor in (0.06, 0.22, 0.62, 1.35)],
            [[factor * capacity] for factor in (0.025, 0.38, 1.8)],
        )
    parameters = {"damping": 0.06 + 0.09 * random.random()}
    return (
        parameters,
        [[0.55, 0.0], [1.45, 0.25], [2.45, -0.15], [-2.65, 0.45]],
        [[3.0, -0.65], [-3.0, 0.85], [2.15, 1.45]],
    )


def _truth_rhs_v312(
    mechanism: MechanismV312,
    state: np.ndarray,
    parameters: dict[str, float],
) -> np.ndarray:
    if mechanism == "gompertz_open_set":
        value = max(float(state[0]), 1e-12)
        return np.asarray([
            parameters["rate"] * value
            * math.log(parameters["capacity"] / value)
        ])
    return np.asarray([
        state[1],
        -math.sin(state[0]) - parameters["damping"] * state[1],
    ])


def _simulate_truth_v312(
    mechanism: MechanismV312,
    initial: list[float],
    times: list[float],
    parameters: dict[str, float],
) -> np.ndarray:
    solution = solve_ivp(
        lambda _time, state: _truth_rhs_v312(mechanism, state, parameters),
        (float(times[0]), float(times[-1])),
        np.asarray(initial, dtype=float),
        t_eval=np.asarray(times, dtype=float),
        rtol=1e-10,
        atol=1e-12,
        max_step=0.01,
    )
    if not solution.success or not np.isfinite(solution.y).all():
        raise RuntimeError("V3.12 hidden truth simulation failed")
    return solution.y.T


def _representation_v312(
    mechanism: MechanismV312,
    dimension: int,
    seed: int,
    representation: RepresentationV312,
) -> tuple[list[int], list[float]]:
    if representation == "anonymous_reference":
        return list(range(dimension)), [1.0] * dimension
    random = Random(seed * 1000003 + MECHANISMS_V312.index(mechanism) * 10007)
    permutation = list(range(dimension))
    random.shuffle(permutation)
    if dimension > 1 and permutation == list(range(dimension)):
        permutation = permutation[1:] + permutation[:1]
    scales = [
        math.exp(random.uniform(math.log(0.45), math.log(2.4)))
        for _ in range(dimension)
    ]
    return permutation, scales


def _to_observed_v312(
    physical: np.ndarray,
    permutation: list[int],
    scales: list[float],
) -> np.ndarray:
    values = physical[:, permutation] * np.asarray(scales)[None, :]
    if not np.isfinite(values).all():
        raise RuntimeError("V3.12 coordinate transform failed")
    return values


def _noisy_v312(
    values: np.ndarray,
    noise_fraction: float,
    seed: int,
    *,
    calibration_failed: bool,
) -> np.ndarray:
    scale = np.maximum(np.std(values, axis=0), 0.05)
    random = np.random.default_rng(seed)
    noisy = values + random.normal(
        0.0, noise_fraction * scale, size=values.shape
    )
    if calibration_failed:
        noisy = noisy + 0.25 * scale[None, :]
    return noisy


def generate_concept_worldpacks_v312(
    private_spec: PrivateConceptWorldPackSpecV312,
    public_protocol: PublicConceptProtocolV312,
    *,
    generated_at: datetime | None = None,
) -> tuple[PublicConceptWorldPackV312, PrivateConceptWorldPackV312]:
    private_spec.assert_sealed()
    public_protocol.assert_sealed()
    if private_spec.public_protocol_hash != public_protocol.protocol_hash:
        raise ValueError("V3.12 generation protocol binding differs")
    times = [
        index * public_protocol.time_step
        for index in range(public_protocol.trajectory_points)
    ]
    private_cases: list[PrivateConceptCaseV312] = []
    for seed_index, seed in enumerate(private_spec.seeds):
        for mechanism_index, mechanism in enumerate(private_spec.mechanisms):
            random = Random(seed * 104729 + mechanism_index * 7919)
            parameters, public_initials, private_initials = _mechanism_setup_v312(
                mechanism, random
            )
            public_physical = [
                _simulate_truth_v312(mechanism, initial, times, parameters)
                for initial in public_initials
            ]
            private_physical = [
                _simulate_truth_v312(mechanism, initial, times, parameters)
                for initial in private_initials
            ]
            dimension = len(public_initials[0])
            pair_id = f"pair_{sha256_value([seed, mechanism])[:16]}"
            calibration_failed = (
                seed_index == private_spec.calibration_failure_seed_index
            )
            for representation_index, representation in enumerate(
                private_spec.representations
            ):
                permutation, scales = _representation_v312(
                    mechanism, dimension, seed, representation
                )
                case_id = f"case_{sha256_value([seed, mechanism, representation])[:16]}"
                trajectories = []
                for trajectory_index, physical in enumerate(public_physical):
                    observed = _to_observed_v312(physical, permutation, scales)
                    observed = _noisy_v312(
                        observed,
                        private_spec.observation_noise_fraction,
                        seed=(
                            seed * 2000003 + mechanism_index * 20011
                            + representation_index * 1009 + trajectory_index * 101
                        ),
                        calibration_failed=calibration_failed,
                    )
                    trajectories.append(AnonymousConceptTrajectoryV312.seal(
                        trajectory_id=f"trajectory_{case_id}_{trajectory_index}",
                        case_id=case_id,
                        times=times,
                        states=observed.tolist(),
                    ))
                public_case = PublicConceptCaseV312.seal(
                    case_id=case_id,
                    state_names=[f"z{index}" for index in range(dimension)],
                    trajectories=trajectories,
                    quality_flags=(
                        ["sensor_calibration_failed"] if calibration_failed else []
                    ),
                )
                private_cases.append(PrivateConceptCaseV312.seal(
                    public_case=public_case,
                    mechanism=mechanism,
                    representation=representation,
                    expected_concept=EXPECTED_CONCEPT_V312[mechanism],
                    hidden_pair_id=pair_id,
                    hidden_seed=seed,
                    hidden_parameters=parameters,
                    observed_to_physical_permutation=permutation,
                    observed_scales=scales,
                    private_probe_initials=[
                        _to_observed_v312(
                            np.asarray(initial)[None, :], permutation, scales
                        )[0].tolist()
                        for initial in private_initials
                    ],
                    private_probe_truths=[
                        _to_observed_v312(physical, permutation, scales).tolist()
                        for physical in private_physical
                    ],
                ))
    public_pack = PublicConceptWorldPackV312.seal(
        public_protocol_hash=public_protocol.protocol_hash,
        cases=[item.public_case for item in private_cases],
    )
    private_pack = PrivateConceptWorldPackV312.seal(
        spec_hash=private_spec.spec_hash,
        public_pack_hash=public_pack.public_pack_hash,
        cases=private_cases,
        generated_at=generated_at or datetime.now(timezone.utc),
    )
    return public_pack, private_pack


class ResidualSignatureV312(StrictModel):
    schema_version: Literal["3.12"] = "3.12"
    case_id: Identifier
    dimension: Annotated[int, Field(ge=1, le=2)]
    baseline_concept: ConceptV312
    baseline_public_score: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    logarithmic_alignment: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    periodic_alignment: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    private_values_used: Literal[False] = False
    signature_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_signature(self) -> "ResidualSignatureV312":
        if self.signature_hash and self.signature_hash != self.content_hash():
            raise ValueError("V3.12 residual signature hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "signature_hash")

    def assert_sealed(self) -> None:
        if not self.signature_hash or self.signature_hash != self.content_hash():
            raise ValueError("V3.12 residual signature is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ResidualSignatureV312":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"signature_hash"}),
            signature_hash=draft.content_hash(),
        )


class ConceptProposalV312(StrictModel):
    schema_version: Literal["3.12"] = "3.12"
    proposal_id: Identifier
    case_id: Identifier
    arm: Literal["fixed_polynomial_baseline", "residual_guided_candidate"]
    concept: ConceptV312
    generator_rank: Annotated[int, Field(ge=1, le=4)]
    evolution_operator: Literal["retain", "add", "replace"]
    rationale: Literal[
        "fixed_budget_control",
        "highest_residual_alignment",
        "grammar_decoy_control",
        "generic_fallback_control",
    ]
    residual_signature_hash: Sha256 | None = None
    private_values_used: Literal[False] = False
    arbitrary_code_used: Literal[False] = False
    proposal_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_proposal(self) -> "ConceptProposalV312":
        if self.arm == "fixed_polynomial_baseline" and self.residual_signature_hash:
            raise ValueError("V3.12 baseline cannot use residual proposal signature")
        if self.arm == "residual_guided_candidate" and not self.residual_signature_hash:
            raise ValueError("V3.12 candidate proposal needs residual signature")
        if self.proposal_hash and self.proposal_hash != self.content_hash():
            raise ValueError("V3.12 proposal hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "proposal_hash")

    def assert_sealed(self) -> None:
        if not self.proposal_hash or self.proposal_hash != self.content_hash():
            raise ValueError("V3.12 proposal is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ConceptProposalV312":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"proposal_hash"}),
            proposal_hash=draft.content_hash(),
        )


class ExpressionModelV312(StrictModel):
    schema_version: Literal["3.12"] = "3.12"
    model_id: Identifier
    case_id: Identifier
    concept: ConceptV312
    dimension: Annotated[int, Field(ge=1, le=2)]
    role_mapping: list[Annotated[int, Field(ge=0, le=1)]] = Field(min_length=1, max_length=2)
    polynomial_degree: Annotated[int, Field(ge=1, le=4)] | None = None
    polynomial_terms: list[PolynomialBasisTermV24]
    coefficients: list[list[Annotated[float, Field(allow_inf_nan=False)]]]
    nonlinear_frequency: Annotated[float, Field(gt=0, allow_inf_nan=False)] | None = None
    parameter_count: Annotated[int, Field(ge=1, le=30)]
    source_trajectory_hashes: list[Sha256] = Field(min_length=2, max_length=2)
    model_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_model(self) -> "ExpressionModelV312":
        if sorted(self.role_mapping) != list(range(self.dimension)):
            raise ValueError("V3.12 role mapping differs")
        if self.concept.startswith("generic_degree_"):
            expected_degree = int(self.concept.rsplit("_", 1)[1])
            if self.polynomial_degree != expected_degree or not self.polynomial_terms:
                raise ValueError("V3.12 generic model degree differs")
        elif self.polynomial_degree is not None or self.polynomial_terms:
            raise ValueError("V3.12 concept model cannot carry polynomial grammar")
        if self.concept == "periodic_restoring_force":
            if self.nonlinear_frequency is None:
                raise ValueError("V3.12 periodic model needs frequency")
        elif self.nonlinear_frequency is not None:
            raise ValueError("V3.12 nonperiodic model cannot carry frequency")
        if self.model_hash and self.model_hash != self.content_hash():
            raise ValueError("V3.12 expression model hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "model_hash")

    def assert_sealed(self) -> None:
        if not self.model_hash or self.model_hash != self.content_hash():
            raise ValueError("V3.12 expression model is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ExpressionModelV312":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"model_hash"}),
            model_hash=draft.content_hash(),
        )


class ConceptAttemptReceiptV312(StrictModel):
    schema_version: Literal["3.12"] = "3.12"
    attempt_id: Identifier
    case_id: Identifier
    arm: Literal["fixed_polynomial_baseline", "residual_guided_candidate"]
    proposal: ConceptProposalV312
    model: ExpressionModelV312 | None
    fit_derivative_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    validation_trajectory_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    challenge_trajectory_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    public_score: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    parameter_count: Annotated[int, Field(ge=0, le=30)]
    valid: bool
    public_evaluator_query_count: Literal[1] = 1
    private_values_used: Literal[False] = False
    attempt_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_attempt(self) -> "ConceptAttemptReceiptV312":
        self.proposal.assert_sealed()
        if self.proposal.case_id != self.case_id or self.proposal.arm != self.arm:
            raise ValueError("V3.12 attempt proposal binding differs")
        if self.model:
            self.model.assert_sealed()
            if self.model.case_id != self.case_id or self.parameter_count != self.model.parameter_count:
                raise ValueError("V3.12 attempt model binding differs")
        elif self.valid or self.parameter_count:
            raise ValueError("V3.12 missing model cannot be valid")
        if self.attempt_hash and self.attempt_hash != self.content_hash():
            raise ValueError("V3.12 attempt hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "attempt_hash")

    def assert_sealed(self) -> None:
        if not self.attempt_hash or self.attempt_hash != self.content_hash():
            raise ValueError("V3.12 attempt is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ConceptAttemptReceiptV312":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"attempt_hash"}),
            attempt_hash=draft.content_hash(),
        )


class ConceptDecisionV312(StrictModel):
    schema_version: Literal["3.12"] = "3.12"
    arm: Literal["fixed_polynomial_baseline", "residual_guided_candidate"]
    selected_attempt_hash: Sha256 | None
    selected_concept: ConceptV312 | None
    selected_public_score: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None
    selection_rule: Literal["minimum_public_ood_score_then_complexity"] = (
        "minimum_public_ood_score_then_complexity"
    )
    private_values_used: Literal[False] = False

    @model_validator(mode="after")
    def validate_decision(self) -> "ConceptDecisionV312":
        present = [
            self.selected_attempt_hash is not None,
            self.selected_concept is not None,
            self.selected_public_score is not None,
        ]
        if any(present) and not all(present):
            raise ValueError("V3.12 decision selection fields differ")
        return self


class ConceptCaseReceiptV312(StrictModel):
    schema_version: Literal["3.12"] = "3.12"
    case_id: Identifier
    public_case_hash: Sha256
    quality_flags: list[Identifier]
    residual_signature: ResidualSignatureV312 | None
    baseline_attempts: list[ConceptAttemptReceiptV312] = Field(max_length=4)
    candidate_attempts: list[ConceptAttemptReceiptV312] = Field(max_length=4)
    baseline_decision: ConceptDecisionV312 | None
    candidate_decision: ConceptDecisionV312 | None
    all_attempts_persisted: Literal[True] = True
    private_values_used: Literal[False] = False
    receipt_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_receipt(self) -> "ConceptCaseReceiptV312":
        if self.quality_flags:
            if (
                self.residual_signature is not None
                or self.baseline_attempts or self.candidate_attempts
                or self.baseline_decision is not None
                or self.candidate_decision is not None
            ):
                raise ValueError("V3.12 quality case must abstain before search")
        else:
            if (
                self.residual_signature is None
                or len(self.baseline_attempts) != 4
                or len(self.candidate_attempts) != 4
                or self.baseline_decision is None
                or self.candidate_decision is None
            ):
                raise ValueError("V3.12 performance case search matrix differs")
            self.residual_signature.assert_sealed()
            for attempt in self.baseline_attempts + self.candidate_attempts:
                attempt.assert_sealed()
        if self.receipt_hash and self.receipt_hash != self.content_hash():
            raise ValueError("V3.12 case receipt hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "receipt_hash")

    def assert_sealed(self) -> None:
        if not self.receipt_hash or self.receipt_hash != self.content_hash():
            raise ValueError("V3.12 case receipt is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ConceptCaseReceiptV312":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"receipt_hash"}),
            receipt_hash=draft.content_hash(),
        )


class ConceptEvolutionBundleV312(StrictModel):
    schema_version: Literal["3.12"] = "3.12"
    bundle_id: Identifier
    public_protocol_hash: Sha256
    public_pack_hash: Sha256
    policy_hash: Sha256
    case_receipts: list[ConceptCaseReceiptV312] = Field(min_length=24, max_length=40)
    created_at: datetime
    bundle_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_bundle(self) -> "ConceptEvolutionBundleV312":
        _assert_timezone(self.created_at, "created_at")
        ids = [item.case_id for item in self.case_receipts]
        if len(ids) != len(set(ids)):
            raise ValueError("V3.12 bundle case ids differ")
        for receipt in self.case_receipts:
            receipt.assert_sealed()
        if self.bundle_hash and self.bundle_hash != self.content_hash():
            raise ValueError("V3.12 bundle hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "bundle_hash")

    def assert_sealed(self) -> None:
        if not self.bundle_hash or self.bundle_hash != self.content_hash():
            raise ValueError("V3.12 bundle is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ConceptEvolutionBundleV312":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"bundle_hash"}),
            bundle_hash=draft.content_hash(),
        )


def _trajectory_values_v312(
    case: PublicConceptCaseV312,
) -> list[np.ndarray]:
    return [np.asarray(item.states, dtype=float) for item in case.trajectories]


def _derivative_v312(values: np.ndarray, protocol: PublicConceptProtocolV312) -> np.ndarray:
    result = savgol_filter(
        values,
        window_length=protocol.savgol_window,
        polyorder=protocol.savgol_polynomial,
        deriv=1,
        delta=protocol.time_step,
        axis=0,
        mode="interp",
    )
    if not np.isfinite(result).all():
        raise FloatingPointError("V3.12 derivative contains nonfinite values")
    return np.asarray(result, dtype=float)


def _normalized_matrix_error_v312(truth: np.ndarray, prediction: np.ndarray) -> float:
    scale = float(np.sqrt(np.mean((truth - np.mean(truth, axis=0)) ** 2)))
    return float(np.sqrt(np.mean((truth - prediction) ** 2)) / max(scale, 1e-8))


def _ridge_v312(matrix: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    gram = matrix.T @ matrix + alpha * np.eye(matrix.shape[1])
    return np.linalg.solve(gram, matrix.T @ target)


def _fit_generic_v312(
    case: PublicConceptCaseV312,
    concept: ConceptV312,
    protocol: PublicConceptProtocolV312,
) -> tuple[ExpressionModelV312, float]:
    degree = int(concept.rsplit("_", 1)[1])
    trajectories = _trajectory_values_v312(case)[:protocol.fit_trajectory_count]
    values = np.vstack(trajectories)
    derivatives = np.vstack([_derivative_v312(item, protocol) for item in trajectories])
    terms = polynomial_basis_terms(case.state_names, degree)
    library = evaluate_polynomial_library(values, terms)
    coefficients = _ridge_v312(library, derivatives, protocol.ridge_alpha)
    prediction = library @ coefficients
    model = ExpressionModelV312.seal(
        model_id=f"model_{case.case_id}_{concept}",
        case_id=case.case_id,
        concept=concept,
        dimension=len(case.state_names),
        role_mapping=list(range(len(case.state_names))),
        polynomial_degree=degree,
        polynomial_terms=terms,
        coefficients=coefficients.T.tolist(),
        parameter_count=int(coefficients.size),
        source_trajectory_hashes=[
            item.trajectory_hash
            for item in case.trajectories[:protocol.fit_trajectory_count]
        ],
    )
    return model, _normalized_matrix_error_v312(derivatives, prediction)


def _fit_scalar_concept_v312(
    case: PublicConceptCaseV312,
    concept: ConceptV312,
    protocol: PublicConceptProtocolV312,
) -> tuple[ExpressionModelV312, float]:
    trajectories = _trajectory_values_v312(case)[:protocol.fit_trajectory_count]
    values = np.vstack(trajectories)[:, 0]
    derivatives = np.vstack([_derivative_v312(item, protocol) for item in trajectories])[:, 0]
    if concept == "logarithmic_rate":
        library = np.column_stack([
            values,
            values * np.log(np.maximum(np.abs(values), 1e-8)),
        ])
    elif concept == "saturating_rate_decoy":
        library = np.column_stack([
            values,
            values / (1.0 + np.abs(values)),
            np.ones_like(values),
        ])
    else:
        library = np.column_stack([np.ones_like(values), values])
    coefficients = _ridge_v312(
        library, derivatives[:, None], protocol.ridge_alpha
    )[:, 0]
    prediction = library @ coefficients
    model = ExpressionModelV312.seal(
        model_id=f"model_{case.case_id}_{concept}",
        case_id=case.case_id,
        concept=concept,
        dimension=1,
        role_mapping=[0],
        polynomial_terms=[],
        coefficients=[coefficients.tolist()],
        parameter_count=len(coefficients),
        source_trajectory_hashes=[
            item.trajectory_hash
            for item in case.trajectories[:protocol.fit_trajectory_count]
        ],
    )
    return model, _normalized_matrix_error_v312(
        derivatives[:, None], prediction[:, None]
    )


def _fit_two_state_concept_v312(
    case: PublicConceptCaseV312,
    concept: ConceptV312,
    protocol: PublicConceptProtocolV312,
) -> tuple[ExpressionModelV312, float]:
    trajectories = _trajectory_values_v312(case)[:protocol.fit_trajectory_count]
    values = np.vstack(trajectories)
    derivatives = np.vstack([_derivative_v312(item, protocol) for item in trajectories])
    best: tuple[float, ExpressionModelV312] | None = None
    for position, velocity in ((0, 1), (1, 0)):
        kinematic = _ridge_v312(
            values[:, velocity, None],
            derivatives[:, position, None],
            protocol.ridge_alpha,
        )[0, 0]
        if concept == "periodic_restoring_force":
            def objective(frequency: float) -> float:
                library = np.column_stack([
                    np.sin(frequency * values[:, position]),
                    values[:, velocity],
                ])
                force = _ridge_v312(
                    library,
                    derivatives[:, velocity, None],
                    protocol.ridge_alpha,
                )[:, 0]
                prediction = np.zeros_like(derivatives)
                prediction[:, position] = kinematic * values[:, velocity]
                prediction[:, velocity] = library @ force
                return _normalized_matrix_error_v312(derivatives, prediction)

            optimized = minimize_scalar(
                objective,
                bounds=(0.2, 2.8),
                method="bounded",
                options={"xatol": 1e-5},
            )
            frequency = float(optimized.x)
            library = np.column_stack([
                np.sin(frequency * values[:, position]),
                values[:, velocity],
            ])
            force = _ridge_v312(
                library,
                derivatives[:, velocity, None],
                protocol.ridge_alpha,
            )[:, 0]
            coefficients = [[float(kinematic), float(force[0]), float(force[1])]]
            parameter_count = 4
            loss = float(optimized.fun)
        elif concept == "kinematic_cubic_decoy":
            library = np.column_stack([
                values[:, position],
                values[:, position] ** 3,
                values[:, velocity],
            ])
            force = _ridge_v312(
                library,
                derivatives[:, velocity, None],
                protocol.ridge_alpha,
            )[:, 0]
            prediction = np.zeros_like(derivatives)
            prediction[:, position] = kinematic * values[:, velocity]
            prediction[:, velocity] = library @ force
            coefficients = [[float(kinematic), *force.tolist()]]
            parameter_count = 4
            frequency = None
            loss = _normalized_matrix_error_v312(derivatives, prediction)
        else:
            coefficients_array = np.zeros((2, 2), dtype=float)
            prediction = np.zeros_like(derivatives)
            for output in range(2):
                local = np.column_stack([
                    np.ones(len(values)), values[:, output]
                ])
                coefficients_array[output] = _ridge_v312(
                    local,
                    derivatives[:, output, None],
                    protocol.ridge_alpha,
                )[:, 0]
                prediction[:, output] = local @ coefficients_array[output]
            coefficients = coefficients_array.tolist()
            parameter_count = 4
            frequency = None
            loss = _normalized_matrix_error_v312(derivatives, prediction)
        model = ExpressionModelV312.seal(
            model_id=f"model_{case.case_id}_{concept}_{position}_{velocity}",
            case_id=case.case_id,
            concept=concept,
            dimension=2,
            role_mapping=[position, velocity],
            polynomial_terms=[],
            coefficients=coefficients,
            nonlinear_frequency=frequency,
            parameter_count=parameter_count,
            source_trajectory_hashes=[
                item.trajectory_hash
                for item in case.trajectories[:protocol.fit_trajectory_count]
            ],
        )
        if best is None or loss < best[0]:
            best = (loss, model)
    if best is None:
        raise RuntimeError("V3.12 two-state role search failed")
    return best[1], best[0]


def _fit_concept_v312(
    case: PublicConceptCaseV312,
    concept: ConceptV312,
    protocol: PublicConceptProtocolV312,
) -> tuple[ExpressionModelV312, float]:
    if concept.startswith("generic_degree_"):
        return _fit_generic_v312(case, concept, protocol)
    if len(case.state_names) == 1:
        return _fit_scalar_concept_v312(case, concept, protocol)
    return _fit_two_state_concept_v312(case, concept, protocol)


def _model_rhs_v312(model: ExpressionModelV312, state: np.ndarray) -> np.ndarray:
    if model.concept.startswith("generic_degree_"):
        library = evaluate_polynomial_library(
            np.asarray(state, dtype=float)[None, :], model.polynomial_terms
        )[0]
        return np.asarray(model.coefficients, dtype=float) @ library
    if model.concept == "logarithmic_rate":
        value = float(state[0])
        features = np.asarray([
            value, value * math.log(max(abs(value), 1e-8))
        ])
        return np.asarray([np.asarray(model.coefficients[0]) @ features])
    if model.concept == "saturating_rate_decoy":
        value = float(state[0])
        features = np.asarray([value, value / (1.0 + abs(value)), 1.0])
        return np.asarray([np.asarray(model.coefficients[0]) @ features])
    if model.concept == "scalar_affine_decoy":
        return np.asarray([
            np.asarray(model.coefficients[0]) @ np.asarray([1.0, state[0]])
        ])
    if model.concept == "uncoupled_linear_decoy":
        coefficients = np.asarray(model.coefficients)
        return np.asarray([
            coefficients[index] @ np.asarray([1.0, state[index]])
            for index in range(2)
        ])
    position, velocity = model.role_mapping
    coefficients = np.asarray(model.coefficients[0])
    result = np.zeros(2, dtype=float)
    result[position] = coefficients[0] * state[velocity]
    if model.concept == "periodic_restoring_force":
        result[velocity] = (
            coefficients[1] * math.sin(model.nonlinear_frequency * state[position])
            + coefficients[2] * state[velocity]
        )
    else:
        result[velocity] = (
            coefficients[1] * state[position]
            + coefficients[2] * state[position] ** 3
            + coefficients[3] * state[velocity]
        )
    return result


def _simulate_model_v312(
    model: ExpressionModelV312,
    initial: list[float] | np.ndarray,
    times: list[float],
) -> np.ndarray:
    solution = solve_ivp(
        lambda _time, state: _model_rhs_v312(model, state),
        (float(times[0]), float(times[-1])),
        np.asarray(initial, dtype=float),
        t_eval=np.asarray(times, dtype=float),
        rtol=1e-7,
        atol=1e-9,
        max_step=0.02,
    )
    if not solution.success or not np.isfinite(solution.y).all():
        raise RuntimeError("V3.12 expression simulation failed")
    return solution.y.T


def _public_trajectory_loss_v312(
    model: ExpressionModelV312,
    trajectory: AnonymousConceptTrajectoryV312,
) -> float:
    truth = np.asarray(trajectory.states, dtype=float)
    prediction = _simulate_model_v312(model, truth[0], trajectory.times)
    return trajectory_nrmse(truth, prediction)


def _correlation_v312(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float).ravel()
    right = np.asarray(right, dtype=float).ravel()
    if np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return 0.0
    return float(min(1.0, abs(np.corrcoef(left, right)[0, 1])))


def _residual_signature_v312(
    case: PublicConceptCaseV312,
    baseline_attempts: list[ConceptAttemptReceiptV312],
    protocol: PublicConceptProtocolV312,
) -> ResidualSignatureV312:
    # A failed baseline is still diagnostic evidence.  Admission validity is a
    # decision constraint, not a precondition for residual analysis.
    fitted = [item for item in baseline_attempts if item.model is not None]
    if not fitted:
        raise ValueError("V3.12 residual analysis needs one fitted baseline model")
    selected = min(
        fitted,
        key=lambda item: (item.public_score, item.parameter_count),
    )
    trajectory = case.trajectories[protocol.validation_trajectory_index]
    values = np.asarray(trajectory.states, dtype=float)
    derivative = _derivative_v312(values, protocol)
    model_derivative = np.asarray([
        _model_rhs_v312(selected.model, row) for row in values
    ])
    residual = derivative - model_derivative
    log_alignment = 0.0
    periodic_alignment = 0.0
    if values.shape[1] == 1:
        feature = values[:, 0] * np.log(np.maximum(np.abs(values[:, 0]), 1e-8))
        log_alignment = _correlation_v312(residual[:, 0], feature)
    else:
        for output, source in itertools.product(range(2), repeat=2):
            for frequency in np.linspace(0.2, 2.8, 14):
                periodic_alignment = max(
                    periodic_alignment,
                    _correlation_v312(
                        residual[:, output], np.sin(frequency * values[:, source])
                    ),
                )
    return ResidualSignatureV312.seal(
        case_id=case.case_id,
        dimension=values.shape[1],
        baseline_concept=selected.proposal.concept,
        baseline_public_score=selected.public_score,
        logarithmic_alignment=log_alignment,
        periodic_alignment=periodic_alignment,
    )


def _proposal_v312(
    case: PublicConceptCaseV312,
    arm: Literal["fixed_polynomial_baseline", "residual_guided_candidate"],
    concept: ConceptV312,
    rank: int,
    signature: ResidualSignatureV312 | None,
) -> ConceptProposalV312:
    if arm == "fixed_polynomial_baseline":
        operator = "retain"
        rationale = "fixed_budget_control"
    elif concept in ("logarithmic_rate", "periodic_restoring_force"):
        operator = "add"
        rationale = "highest_residual_alignment"
    elif concept == "generic_degree_3":
        operator = "retain"
        rationale = "generic_fallback_control"
    else:
        operator = "replace"
        rationale = "grammar_decoy_control"
    return ConceptProposalV312.seal(
        proposal_id=f"proposal_{case.case_id}_{arm}_{rank}",
        case_id=case.case_id,
        arm=arm,
        concept=concept,
        generator_rank=rank,
        evolution_operator=operator,
        rationale=rationale,
        residual_signature_hash=(signature.signature_hash if signature else None),
    )


def _attempt_v312(
    case: PublicConceptCaseV312,
    arm: Literal["fixed_polynomial_baseline", "residual_guided_candidate"],
    concept: ConceptV312,
    rank: int,
    signature: ResidualSignatureV312 | None,
    protocol: PublicConceptProtocolV312,
) -> ConceptAttemptReceiptV312:
    proposal = _proposal_v312(case, arm, concept, rank, signature)
    try:
        model, fit_loss = _fit_concept_v312(case, concept, protocol)
        validation_loss = _public_trajectory_loss_v312(
            model, case.trajectories[protocol.validation_trajectory_index]
        )
        challenge_loss = _public_trajectory_loss_v312(
            model, case.trajectories[protocol.challenge_trajectory_index]
        )
        score = max(validation_loss, challenge_loss) + (
            protocol.complexity_penalty * model.parameter_count
        )
        valid = bool(np.isfinite(score) and score <= protocol.maximum_public_score)
        parameter_count = model.parameter_count
    except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError):
        model = None
        fit_loss = protocol.unresolved_loss
        validation_loss = protocol.unresolved_loss
        challenge_loss = protocol.unresolved_loss
        score = protocol.unresolved_loss
        valid = False
        parameter_count = 0
    return ConceptAttemptReceiptV312.seal(
        attempt_id=f"attempt_{case.case_id}_{arm}_{rank}",
        case_id=case.case_id,
        arm=arm,
        proposal=proposal,
        model=model,
        fit_derivative_loss=fit_loss,
        validation_trajectory_loss=validation_loss,
        challenge_trajectory_loss=challenge_loss,
        public_score=score,
        parameter_count=parameter_count,
        valid=valid,
    )


def _select_attempt_v312(
    arm: Literal["fixed_polynomial_baseline", "residual_guided_candidate"],
    attempts: list[ConceptAttemptReceiptV312],
) -> ConceptDecisionV312:
    # The control arm must remain observable even when it misses the candidate
    # admission threshold; otherwise its private loss is replaced by the
    # unresolved penalty and the treatment effect is inflated.  The concept
    # arm, in contrast, must pass that public threshold before adjudication.
    eligible = (
        [item for item in attempts if item.model is not None]
        if arm == "fixed_polynomial_baseline"
        else [item for item in attempts if item.valid]
    )
    if not eligible:
        return ConceptDecisionV312(
            arm=arm,
            selected_attempt_hash=None,
            selected_concept=None,
            selected_public_score=None,
        )
    selected = min(
        eligible,
        key=lambda item: (
            item.public_score,
            item.parameter_count,
            item.proposal.generator_rank,
        ),
    )
    return ConceptDecisionV312(
        arm=arm,
        selected_attempt_hash=selected.attempt_hash,
        selected_concept=selected.proposal.concept,
        selected_public_score=selected.public_score,
    )


def _execute_case_v312(
    case: PublicConceptCaseV312,
    policy: ConceptEvolutionPolicyV312,
    protocol: PublicConceptProtocolV312,
) -> ConceptCaseReceiptV312:
    case.assert_sealed()
    if case.quality_flags:
        return ConceptCaseReceiptV312.seal(
            case_id=case.case_id,
            public_case_hash=case.public_hash,
            quality_flags=case.quality_flags,
            residual_signature=None,
            baseline_attempts=[],
            candidate_attempts=[],
            baseline_decision=None,
            candidate_decision=None,
        )
    baseline_attempts = [
        _attempt_v312(
            case, "fixed_polynomial_baseline", concept, index, None, protocol
        )
        for index, concept in enumerate(policy.baseline_concepts, start=1)
    ]
    signature = _residual_signature_v312(case, baseline_attempts, protocol)
    concepts = policy.candidate_concepts_by_dimension[len(case.state_names)]
    target = (
        "logarithmic_rate" if len(case.state_names) == 1
        else "periodic_restoring_force"
    )
    ordered = [target] + [item for item in concepts if item != target]
    candidate_attempts = [
        _attempt_v312(
            case, "residual_guided_candidate", concept, index, signature, protocol
        )
        for index, concept in enumerate(ordered, start=1)
    ]
    return ConceptCaseReceiptV312.seal(
        case_id=case.case_id,
        public_case_hash=case.public_hash,
        quality_flags=case.quality_flags,
        residual_signature=signature,
        baseline_attempts=baseline_attempts,
        candidate_attempts=candidate_attempts,
        baseline_decision=_select_attempt_v312(
            "fixed_polynomial_baseline", baseline_attempts
        ),
        candidate_decision=_select_attempt_v312(
            "residual_guided_candidate", candidate_attempts
        ),
    )


def execute_concept_evolution_v312(
    public_protocol: PublicConceptProtocolV312,
    public_pack: PublicConceptWorldPackV312,
    policy: ConceptEvolutionPolicyV312,
    *,
    executed_at: datetime,
) -> ConceptEvolutionBundleV312:
    for item in (public_protocol, public_pack, policy):
        item.assert_sealed()
    if (
        public_pack.public_protocol_hash != public_protocol.protocol_hash
        or policy.policy_hash != public_protocol.policy_hash
    ):
        raise ValueError("V3.12 public execution binding differs")
    return ConceptEvolutionBundleV312.seal(
        bundle_id="open_set_concept_evolution_bundle_v312",
        public_protocol_hash=public_protocol.protocol_hash,
        public_pack_hash=public_pack.public_pack_hash,
        policy_hash=policy.policy_hash,
        case_receipts=[
            _execute_case_v312(case, policy, public_protocol)
            for case in public_pack.cases
        ],
        created_at=executed_at,
    )


class PrivateConceptCaseResultV312(StrictModel):
    schema_version: Literal["3.12"] = "3.12"
    case_id: Identifier
    hidden_pair_id: Identifier
    mechanism: MechanismV312
    representation: RepresentationV312
    expected_concept: ConceptV312
    baseline_concept: ConceptV312 | None
    candidate_concept: ConceptV312 | None
    baseline_target_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    candidate_target_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    candidate_improvement: Annotated[float, Field(allow_inf_nan=False)]
    concept_correct: bool
    material_negative_transfer: bool
    private_values_visible_to_generator: Literal[False] = False

    @model_validator(mode="after")
    def validate_result(self) -> "PrivateConceptCaseResultV312":
        if not math.isclose(
            self.candidate_improvement,
            self.baseline_target_loss - self.candidate_target_loss,
            abs_tol=1e-12,
        ):
            raise ValueError("V3.12 case improvement does not recompute")
        return self


class ConceptRepresentationPairResultV312(StrictModel):
    schema_version: Literal["3.12"] = "3.12"
    hidden_pair_id: Identifier
    mechanism: MechanismV312
    reference_case_id: Identifier
    transformed_case_id: Identifier
    reference_concept: ConceptV312 | None
    transformed_concept: ConceptV312 | None
    concept_consistent: bool
    reference_target_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    transformed_target_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    absolute_target_loss_difference: Annotated[float, Field(ge=0, allow_inf_nan=False)]

    @model_validator(mode="after")
    def validate_pair(self) -> "ConceptRepresentationPairResultV312":
        if not math.isclose(
            self.absolute_target_loss_difference,
            abs(self.reference_target_loss - self.transformed_target_loss),
            abs_tol=1e-12,
        ):
            raise ValueError("V3.12 pair loss difference does not recompute")
        return self


class ConceptLedgerEntryV312(StrictModel):
    schema_version: Literal["3.12"] = "3.12"
    concept: ConceptV312
    status: Literal[
        "development_candidate",
        "admitted_private_confirmation",
        "rejected_private_confirmation",
    ]
    supporting_case_count: Annotated[int, Field(ge=0)]
    correct_case_count: Annotated[int, Field(ge=0)]
    private_mean_improvement: Annotated[float, Field(allow_inf_nan=False)]
    public_score_used_for_admission: Literal[False] = False
    private_confirmation_required: Literal[True] = True


class ConceptEvolutionReportV312(StrictModel):
    schema_version: Literal["3.12"] = "3.12"
    report_id: Identifier
    phase: Literal["development", "confirmation"]
    private_spec_hash: Sha256
    public_protocol_hash: Sha256
    public_pack_hash: Sha256
    private_pack_hash: Sha256
    bundle_hash: Sha256
    lineage_receipt_hash: Sha256
    source_v311_report_hash: Sha256
    case_results: list[PrivateConceptCaseResultV312]
    pair_results: list[ConceptRepresentationPairResultV312]
    concept_ledger: list[ConceptLedgerEntryV312]
    performance_case_count: Annotated[int, Field(ge=1)]
    quality_case_count: Annotated[int, Field(ge=0)]
    baseline_expression_evaluation_count: Annotated[int, Field(ge=1)]
    candidate_expression_evaluation_count: Annotated[int, Field(ge=1)]
    all_attempts_persisted: bool
    equal_evaluation_budget: bool
    baseline_coverage: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    candidate_coverage: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    baseline_mean_target_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    candidate_mean_target_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    paired_mean_improvement: Annotated[float, Field(allow_inf_nan=False)]
    paired_improvement_ci_lower: Annotated[float, Field(allow_inf_nan=False)]
    paired_improvement_ci_upper: Annotated[float, Field(allow_inf_nan=False)]
    concept_recovery_accuracy: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    pair_concept_consistency: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    maximum_pair_loss_difference: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    mean_loss_by_mechanism_baseline: dict[MechanismV312, Annotated[float, Field(ge=0, allow_inf_nan=False)]]
    mean_loss_by_mechanism_candidate: dict[MechanismV312, Annotated[float, Field(ge=0, allow_inf_nan=False)]]
    mean_loss_by_representation_candidate: dict[RepresentationV312, Annotated[float, Field(ge=0, allow_inf_nan=False)]]
    candidate_selection_counts: dict[ConceptV312, Annotated[int, Field(ge=0)]]
    material_negative_transfer_count: Annotated[int, Field(ge=0)]
    material_negative_transfer_upper_95: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    public_execution_private_blind: bool
    gates: dict[Identifier, bool]
    ready_for_concept_admission: bool
    task_router_permitted: Literal[False] = False
    model_qualification_permitted: Literal[False] = False
    real_world_execution_permitted: Literal[False] = False
    status: Literal[
        "concept_evolution_development_diagnostic_v312",
        "open_set_concepts_admitted_v312",
        "open_set_concept_evolution_refuted_v312",
    ]
    created_at: datetime
    report_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_report(self) -> "ConceptEvolutionReportV312":
        _assert_timezone(self.created_at, "created_at")
        expected_ready = self.phase == "confirmation" and all(self.gates.values())
        if self.phase == "development":
            expected_status = "concept_evolution_development_diagnostic_v312"
        elif expected_ready:
            expected_status = "open_set_concepts_admitted_v312"
        else:
            expected_status = "open_set_concept_evolution_refuted_v312"
        if self.ready_for_concept_admission != expected_ready or self.status != expected_status:
            raise ValueError("V3.12 report status differs from gates")
        if set(self.mean_loss_by_mechanism_candidate) != set(MECHANISMS_V312):
            raise ValueError("V3.12 mechanism report incomplete")
        if set(self.mean_loss_by_representation_candidate) != set(REPRESENTATIONS_V312):
            raise ValueError("V3.12 representation report incomplete")
        if self.report_hash and self.report_hash != self.content_hash():
            raise ValueError("V3.12 report hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "report_hash")

    def assert_sealed(self) -> None:
        if not self.report_hash or self.report_hash != self.content_hash():
            raise ValueError("V3.12 report is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ConceptEvolutionReportV312":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"report_hash"}),
            report_hash=draft.content_hash(),
        )


def _selected_attempt_v312(
    receipt: ConceptCaseReceiptV312,
    decision: ConceptDecisionV312,
) -> ConceptAttemptReceiptV312 | None:
    if decision.selected_attempt_hash is None:
        return None
    attempts = (
        receipt.baseline_attempts
        if decision.arm == "fixed_polynomial_baseline"
        else receipt.candidate_attempts
    )
    matches = [
        item for item in attempts
        if item.attempt_hash == decision.selected_attempt_hash
    ]
    if len(matches) != 1:
        raise ValueError("V3.12 selected attempt binding differs")
    return matches[0]


def _private_loss_v312(
    private_case: PrivateConceptCaseV312,
    attempt: ConceptAttemptReceiptV312 | None,
    protocol: PublicConceptProtocolV312,
) -> float:
    if attempt is None or attempt.model is None:
        return protocol.unresolved_loss
    times = private_case.public_case.trajectories[0].times
    losses = []
    for initial, truth in zip(
        private_case.private_probe_initials,
        private_case.private_probe_truths,
        strict=True,
    ):
        try:
            prediction = _simulate_model_v312(attempt.model, initial, times)
            losses.append(trajectory_nrmse(truth, prediction))
        except RuntimeError:
            losses.append(protocol.unresolved_loss)
    return float(np.mean(losses))


def _bootstrap_ci_v312(
    values: np.ndarray,
    spec: PrivateConceptWorldPackSpecV312,
) -> tuple[float, float]:
    random = np.random.default_rng(spec.bootstrap_seed)
    indices = random.integers(
        0, len(values), size=(spec.bootstrap_replicates, len(values))
    )
    means = np.mean(values[indices], axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def evaluate_concept_evolution_v312(
    private_spec: PrivateConceptWorldPackSpecV312,
    public_protocol: PublicConceptProtocolV312,
    lineage: VerifiedLineageReceiptV312,
    private_pack: PrivateConceptWorldPackV312,
    bundle: ConceptEvolutionBundleV312,
    *,
    evaluated_at: datetime,
) -> ConceptEvolutionReportV312:
    for item in (private_spec, public_protocol, lineage, private_pack, bundle):
        item.assert_sealed()
    if (
        private_pack.spec_hash != private_spec.spec_hash
        or private_pack.public_pack_hash != bundle.public_pack_hash
        or bundle.public_protocol_hash != public_protocol.protocol_hash
        or lineage.receipt_hash != private_spec.lineage_receipt_hash
    ):
        raise ValueError("V3.12 evaluator binding differs")
    receipts = {item.case_id: item for item in bundle.case_receipts}
    results: list[PrivateConceptCaseResultV312] = []
    quality_count = 0
    baseline_evaluations = 0
    candidate_evaluations = 0
    all_persisted = True
    private_blind = True
    baseline_by_mechanism: dict[str, list[float]] = defaultdict(list)
    candidate_by_mechanism: dict[str, list[float]] = defaultdict(list)
    candidate_by_representation: dict[str, list[float]] = defaultdict(list)
    selection_counts: dict[str, int] = defaultdict(int)
    for private_case in private_pack.cases:
        receipt = receipts[private_case.public_case.case_id]
        if receipt.quality_flags != private_case.public_case.quality_flags:
            private_blind = False
        if receipt.quality_flags:
            quality_count += 1
            continue
        baseline_evaluations += sum(
            item.public_evaluator_query_count for item in receipt.baseline_attempts
        )
        candidate_evaluations += sum(
            item.public_evaluator_query_count for item in receipt.candidate_attempts
        )
        all_attempts = receipt.baseline_attempts + receipt.candidate_attempts
        all_persisted = all_persisted and (
            len(all_attempts) == 8
            and len({item.attempt_hash for item in all_attempts}) == 8
            and receipt.all_attempts_persisted
        )
        private_blind = private_blind and not receipt.private_values_used and all(
            not item.private_values_used
            and not item.proposal.private_values_used
            and not item.proposal.arbitrary_code_used
            for item in all_attempts
        )
        baseline_attempt = _selected_attempt_v312(
            receipt, receipt.baseline_decision
        )
        candidate_attempt = _selected_attempt_v312(
            receipt, receipt.candidate_decision
        )
        baseline_loss = _private_loss_v312(
            private_case, baseline_attempt, public_protocol
        )
        candidate_loss = _private_loss_v312(
            private_case, candidate_attempt, public_protocol
        )
        candidate_concept = receipt.candidate_decision.selected_concept
        if candidate_concept:
            selection_counts[candidate_concept] += 1
        baseline_by_mechanism[private_case.mechanism].append(baseline_loss)
        candidate_by_mechanism[private_case.mechanism].append(candidate_loss)
        candidate_by_representation[private_case.representation].append(candidate_loss)
        results.append(PrivateConceptCaseResultV312(
            case_id=private_case.public_case.case_id,
            hidden_pair_id=private_case.hidden_pair_id,
            mechanism=private_case.mechanism,
            representation=private_case.representation,
            expected_concept=private_case.expected_concept,
            baseline_concept=(
                receipt.baseline_decision.selected_concept
                if receipt.baseline_decision else None
            ),
            candidate_concept=candidate_concept,
            baseline_target_loss=baseline_loss,
            candidate_target_loss=candidate_loss,
            candidate_improvement=baseline_loss - candidate_loss,
            concept_correct=(candidate_concept == private_case.expected_concept),
            material_negative_transfer=(
                candidate_loss - baseline_loss
                > private_spec.material_negative_transfer
            ),
        ))
    by_pair: dict[str, list[PrivateConceptCaseResultV312]] = defaultdict(list)
    for result in results:
        by_pair[result.hidden_pair_id].append(result)
    pairs: list[ConceptRepresentationPairResultV312] = []
    for pair_id, members in by_pair.items():
        if len(members) != 2:
            raise ValueError("V3.12 representation pair incomplete")
        reference = next(
            item for item in members
            if item.representation == "anonymous_reference"
        )
        transformed = next(
            item for item in members
            if item.representation == "anonymous_scaled_permuted"
        )
        pairs.append(ConceptRepresentationPairResultV312(
            hidden_pair_id=pair_id,
            mechanism=reference.mechanism,
            reference_case_id=reference.case_id,
            transformed_case_id=transformed.case_id,
            reference_concept=reference.candidate_concept,
            transformed_concept=transformed.candidate_concept,
            concept_consistent=(
                reference.candidate_concept == transformed.candidate_concept
            ),
            reference_target_loss=reference.candidate_target_loss,
            transformed_target_loss=transformed.candidate_target_loss,
            absolute_target_loss_difference=abs(
                reference.candidate_target_loss
                - transformed.candidate_target_loss
            ),
        ))
    baseline_losses = np.asarray([item.baseline_target_loss for item in results])
    candidate_losses = np.asarray([item.candidate_target_loss for item in results])
    improvements = baseline_losses - candidate_losses
    ci_lower, ci_upper = _bootstrap_ci_v312(improvements, private_spec)
    negatives = sum(item.material_negative_transfer for item in results)
    negative_upper = (
        float(beta.ppf(0.95, negatives + 1, len(results) - negatives))
        if len(results) > negatives else 1.0
    )
    mechanism_baseline = {
        mechanism: float(np.mean(baseline_by_mechanism[mechanism]))
        for mechanism in MECHANISMS_V312
    }
    mechanism_candidate = {
        mechanism: float(np.mean(candidate_by_mechanism[mechanism]))
        for mechanism in MECHANISMS_V312
    }
    representation_candidate = {
        representation: float(np.mean(candidate_by_representation[representation]))
        for representation in REPRESENTATIONS_V312
    }
    recovery_accuracy = float(np.mean([item.concept_correct for item in results]))
    pair_consistency = float(np.mean([item.concept_consistent for item in pairs]))
    max_pair_difference = max(item.absolute_target_loss_difference for item in pairs)
    ledger: list[ConceptLedgerEntryV312] = []
    for concept in ("logarithmic_rate", "periodic_restoring_force"):
        relevant = [item for item in results if item.expected_concept == concept]
        correct = sum(item.candidate_concept == concept for item in relevant)
        improvement = float(np.mean([item.candidate_improvement for item in relevant]))
        eligible = (
            correct / len(relevant) >= private_spec.minimum_concept_recovery_accuracy
            and improvement >= -private_spec.maximum_mechanism_regression
        )
        if private_spec.phase == "development":
            status = "development_candidate"
        elif eligible:
            status = "admitted_private_confirmation"
        else:
            status = "rejected_private_confirmation"
        ledger.append(ConceptLedgerEntryV312(
            concept=concept,
            status=status,
            supporting_case_count=len(relevant),
            correct_case_count=correct,
            private_mean_improvement=improvement,
        ))
    decoys: tuple[ConceptV312, ...] = (
        "saturating_rate_decoy", "scalar_affine_decoy",
        "kinematic_cubic_decoy", "uncoupled_linear_decoy",
    )
    for concept in decoys:
        selected_count = selection_counts.get(concept, 0)
        ledger.append(ConceptLedgerEntryV312(
            concept=concept,
            status=(
                "development_candidate" if private_spec.phase == "development"
                else "rejected_private_confirmation"
            ),
            supporting_case_count=selected_count,
            correct_case_count=0,
            private_mean_improvement=0.0,
        ))
    equal_budget = (
        baseline_evaluations == candidate_evaluations
        == len(results) * 4
    )
    expected_ledger = (
        all(item.status == "development_candidate" for item in ledger)
        if private_spec.phase == "development"
        else (
            all(
                item.status == "admitted_private_confirmation"
                for item in ledger[:2]
            )
            and all(
                item.status == "rejected_private_confirmation"
                for item in ledger[2:]
            )
        )
    )
    gates = {
        "public_quality_partition_complete": (
            quality_count == private_spec.expected_quality_case_count
        ),
        "equal_expression_evaluation_budget": equal_budget,
        "all_attempts_persisted": all_persisted,
        "public_execution_private_blind": private_blind,
        "candidate_coverage": (
            sum(item.candidate_concept is not None for item in results) / len(results)
            >= private_spec.minimum_concept_recovery_accuracy
        ),
        "concept_recovery_accuracy": (
            recovery_accuracy >= private_spec.minimum_concept_recovery_accuracy
        ),
        "paired_concept_consistency": (
            pair_consistency >= private_spec.minimum_pair_concept_consistency
        ),
        "paired_prediction_invariance": (
            max_pair_difference <= private_spec.maximum_pair_loss_difference
        ),
        "paired_improvement_ci_lower_positive": ci_lower > 0.0,
        "mechanism_non_regression": all(
            mechanism_candidate[mechanism]
            <= mechanism_baseline[mechanism]
            + private_spec.maximum_mechanism_regression
            for mechanism in MECHANISMS_V312
        ),
        "transformed_representation_non_regression": (
            representation_candidate["anonymous_scaled_permuted"]
            <= representation_candidate["anonymous_reference"]
            + private_spec.maximum_transformed_representation_regression
        ),
        "material_negative_transfer_controlled": (
            negative_upper <= private_spec.maximum_negative_transfer_upper_95
        ),
        "concept_ledger_private_adjudication": expected_ledger,
        "no_decoy_concept_selected": all(
            selection_counts.get(concept, 0) == 0 for concept in decoys
        ),
        "no_task_router_or_real_world_execution": True,
    }
    ready = private_spec.phase == "confirmation" and all(gates.values())
    if private_spec.phase == "development":
        status = "concept_evolution_development_diagnostic_v312"
    elif ready:
        status = "open_set_concepts_admitted_v312"
    else:
        status = "open_set_concept_evolution_refuted_v312"
    all_concepts = list(ConceptV312.__args__)
    return ConceptEvolutionReportV312.seal(
        report_id=f"report_{private_spec.experiment_id}",
        phase=private_spec.phase,
        private_spec_hash=private_spec.spec_hash,
        public_protocol_hash=public_protocol.protocol_hash,
        public_pack_hash=private_pack.public_pack_hash,
        private_pack_hash=private_pack.pack_hash,
        bundle_hash=bundle.bundle_hash,
        lineage_receipt_hash=lineage.receipt_hash,
        source_v311_report_hash=lineage.source_report_hash,
        case_results=results,
        pair_results=pairs,
        concept_ledger=ledger,
        performance_case_count=len(results),
        quality_case_count=quality_count,
        baseline_expression_evaluation_count=baseline_evaluations,
        candidate_expression_evaluation_count=candidate_evaluations,
        all_attempts_persisted=all_persisted,
        equal_evaluation_budget=equal_budget,
        baseline_coverage=sum(item.baseline_concept is not None for item in results) / len(results),
        candidate_coverage=sum(item.candidate_concept is not None for item in results) / len(results),
        baseline_mean_target_loss=float(np.mean(baseline_losses)),
        candidate_mean_target_loss=float(np.mean(candidate_losses)),
        paired_mean_improvement=float(np.mean(improvements)),
        paired_improvement_ci_lower=ci_lower,
        paired_improvement_ci_upper=ci_upper,
        concept_recovery_accuracy=recovery_accuracy,
        pair_concept_consistency=pair_consistency,
        maximum_pair_loss_difference=max_pair_difference,
        mean_loss_by_mechanism_baseline=mechanism_baseline,
        mean_loss_by_mechanism_candidate=mechanism_candidate,
        mean_loss_by_representation_candidate=representation_candidate,
        candidate_selection_counts={
            concept: selection_counts.get(concept, 0) for concept in all_concepts
        },
        material_negative_transfer_count=negatives,
        material_negative_transfer_upper_95=negative_upper,
        public_execution_private_blind=private_blind,
        gates=gates,
        ready_for_concept_admission=ready,
        status=status,
        created_at=evaluated_at,
    )


class ConceptEvolutionManifestV312(StrictModel):
    schema_version: Literal["3.12"] = "3.12"
    manifest_id: Identifier
    run_id: Identifier
    phase: Literal["development", "confirmation"]
    artifact_refs: list[ArtifactRef] = Field(min_length=9, max_length=9)
    terminal_status: Literal[
        "concept_evolution_development_diagnostic_v312",
        "open_set_concepts_admitted_v312",
        "open_set_concept_evolution_refuted_v312",
    ]
    created_at: datetime
    manifest_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_manifest(self) -> "ConceptEvolutionManifestV312":
        _assert_timezone(self.created_at, "created_at")
        if self.manifest_hash and self.manifest_hash != self.content_hash():
            raise ValueError("V3.12 manifest hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "manifest_hash")

    def assert_sealed(self) -> None:
        if not self.manifest_hash or self.manifest_hash != self.content_hash():
            raise ValueError("V3.12 manifest is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ConceptEvolutionManifestV312":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"manifest_hash"}),
            manifest_hash=draft.content_hash(),
        )


@dataclass(frozen=True)
class ConceptEvolutionOutcomeV312:
    store: RunStore
    public_pack: PublicConceptWorldPackV312
    private_pack: PrivateConceptWorldPackV312
    bundle: ConceptEvolutionBundleV312
    report: ConceptEvolutionReportV312
    manifest: ConceptEvolutionManifestV312


def _load_development_report_v312(
    run_directory: str | Path,
    *,
    source_run_directory: str | Path,
) -> ConceptEvolutionReportV312:
    if not verify_concept_evolution_run_v312(
        run_directory,
        source_run_directory=source_run_directory,
        development_run_directory=None,
    ):
        raise ValueError("V3.12 development run did not independently verify")
    store = RunStore.open_existing(run_directory)
    report = _load_one(
        store, _committed_refs(store), "concept_evolution_report_v312",
        ConceptEvolutionReportV312,
    )
    if report.phase != "development":
        raise ValueError("V3.12 development lineage phase differs")
    return report


def run_concept_evolution_worldpack_v312(
    output_root: str | Path,
    *,
    source_run_directory: str | Path,
    development_run_directory: str | Path | None = None,
    evidence: ConceptMethodEvidenceV312,
    lineage: VerifiedLineageReceiptV312,
    policy: ConceptEvolutionPolicyV312,
    public_protocol: PublicConceptProtocolV312,
    private_spec: PrivateConceptWorldPackSpecV312,
    evaluated_at: datetime | None = None,
    run_id: str | None = None,
) -> ConceptEvolutionOutcomeV312:
    if not verify_v311_lineage_receipt_v312(lineage, source_run_directory):
        raise ValueError("V3.12 source lineage receipt failed quick verification")
    development_report = None
    if private_spec.phase == "confirmation":
        if development_run_directory is None:
            raise ValueError("V3.12 confirmation requires development run")
        development_report = _load_development_report_v312(
            development_run_directory,
            source_run_directory=source_run_directory,
        )
    for item in (evidence, lineage, policy, public_protocol, private_spec):
        item.assert_sealed()
    if (
        evidence.evidence_hash != policy.method_evidence_hash
        or evidence.evidence_hash != public_protocol.method_evidence_hash
        or lineage.receipt_hash != policy.lineage_receipt_hash
        or lineage.receipt_hash != public_protocol.lineage_receipt_hash
        or lineage.receipt_hash != private_spec.lineage_receipt_hash
        or policy.policy_hash != public_protocol.policy_hash
        or public_protocol.protocol_hash != private_spec.public_protocol_hash
        or private_spec.frozen_at < public_protocol.frozen_at
        or (
            private_spec.phase == "confirmation"
            and (
                development_report.report_hash != private_spec.development_report_hash
                or private_spec.frozen_at < development_report.created_at
            )
        )
    ):
        raise ValueError("V3.12 frozen lineage binding differs")
    at = evaluated_at or datetime.now(timezone.utc)
    wall_now = datetime.now(timezone.utc)
    if at < private_spec.frozen_at:
        raise ValueError("V3.12 evaluation predates private spec")
    if (
        private_spec.frozen_at > wall_now + timedelta(minutes=5)
        or at > wall_now + timedelta(minutes=5)
    ):
        raise ValueError("V3.12 audit timestamp is implausibly in the future")
    store = RunStore(
        output_root,
        run_id=run_id or f"open-set-concept-v312-{uuid4().hex[:10]}",
    )
    refs = [
        store.put_artifact("concept_method_evidence_v312", evidence),
        store.put_artifact("verified_lineage_receipt_v312", lineage),
        store.put_artifact("concept_evolution_policy_v312", policy),
        store.put_artifact("public_concept_protocol_v312", public_protocol),
        store.put_artifact("private_concept_worldpack_spec_v312", private_spec),
    ]
    store.emit("concept_evolution_v312_protocol_frozen_before_private_pack", {
        "phase": private_spec.phase,
        "private_spec_hash": private_spec.spec_hash,
        "development_report_hash": private_spec.development_report_hash,
        "source_lineage_receipt_hash": lineage.receipt_hash,
        "private_pack_not_passed_to_generator": True,
        "public_score_cannot_admit_concept": True,
    })
    public_pack, private_pack = generate_concept_worldpacks_v312(
        private_spec, public_protocol, generated_at=at
    )
    bundle = execute_concept_evolution_v312(
        public_protocol, public_pack, policy, executed_at=at
    )
    report = evaluate_concept_evolution_v312(
        private_spec, public_protocol, lineage, private_pack, bundle,
        evaluated_at=at,
    )
    refs.extend([
        store.put_artifact("public_concept_worldpack_v312", public_pack),
        store.put_artifact("private_concept_worldpack_v312", private_pack),
        store.put_artifact("concept_evolution_bundle_v312", bundle),
        store.put_artifact("concept_evolution_report_v312", report),
    ])
    manifest = ConceptEvolutionManifestV312.seal(
        manifest_id=f"manifest_{store.run_id}",
        run_id=store.run_id,
        phase=private_spec.phase,
        artifact_refs=refs,
        terminal_status=report.status,
        created_at=at,
    )
    manifest_ref = store.put_artifact("concept_evolution_manifest_v312", manifest)
    store.emit("concept_evolution_v312_worldpack_adjudicated", {
        "manifest_ref": manifest_ref.model_dump(mode="json")
    })
    if not verify_concept_evolution_run_v312(
        store.run_directory,
        source_run_directory=source_run_directory,
        development_run_directory=development_run_directory,
    ):
        raise RuntimeError("V3.12 run failed independent verification")
    return ConceptEvolutionOutcomeV312(
        store, public_pack, private_pack, bundle, report, manifest
    )


def verify_concept_evolution_run_v312(
    run_directory: str | Path,
    *,
    source_run_directory: str | Path,
    development_run_directory: str | Path | None = None,
) -> bool:
    try:
        store = RunStore.open_existing(run_directory)
        if not store.verify_event_chain():
            return False
        events = [
            json.loads(line)
            for line in store.event_path.read_text(encoding="utf-8").splitlines()
        ]
        refs = _committed_refs(store)
        if len(refs) != 10:
            return False
        for ref in refs:
            store.load_artifact(ref)
        evidence = _load_one(
            store, refs, "concept_method_evidence_v312",
            ConceptMethodEvidenceV312,
        )
        lineage = _load_one(
            store, refs, "verified_lineage_receipt_v312",
            VerifiedLineageReceiptV312,
        )
        policy = _load_one(
            store, refs, "concept_evolution_policy_v312",
            ConceptEvolutionPolicyV312,
        )
        protocol = _load_one(
            store, refs, "public_concept_protocol_v312",
            PublicConceptProtocolV312,
        )
        spec = _load_one(
            store, refs, "private_concept_worldpack_spec_v312",
            PrivateConceptWorldPackSpecV312,
        )
        public_pack = _load_one(
            store, refs, "public_concept_worldpack_v312",
            PublicConceptWorldPackV312,
        )
        private_pack = _load_one(
            store, refs, "private_concept_worldpack_v312",
            PrivateConceptWorldPackV312,
        )
        bundle = _load_one(
            store, refs, "concept_evolution_bundle_v312",
            ConceptEvolutionBundleV312,
        )
        report = _load_one(
            store, refs, "concept_evolution_report_v312",
            ConceptEvolutionReportV312,
        )
        manifest = _load_one(
            store, refs, "concept_evolution_manifest_v312",
            ConceptEvolutionManifestV312,
        )
        for item in (
            evidence, lineage, policy, protocol, spec,
            public_pack, private_pack, bundle, report, manifest,
        ):
            item.assert_sealed()
        wall_now = datetime.now(timezone.utc)
        if (
            spec.frozen_at < protocol.frozen_at
            or report.created_at < spec.frozen_at
            or spec.frozen_at > wall_now + timedelta(minutes=5)
            or report.created_at > wall_now + timedelta(minutes=5)
        ):
            return False
        development_report = None
        if spec.phase == "confirmation":
            if development_run_directory is None:
                return False
            if Path(development_run_directory).resolve() == Path(run_directory).resolve():
                return False
            development_report = _load_development_report_v312(
                development_run_directory,
                source_run_directory=source_run_directory,
            )
        if (
            not verify_v311_lineage_receipt_v312(lineage, source_run_directory)
            or evidence.evidence_hash != policy.method_evidence_hash
            or evidence.evidence_hash != protocol.method_evidence_hash
            or lineage.receipt_hash != policy.lineage_receipt_hash
            or lineage.receipt_hash != protocol.lineage_receipt_hash
            or lineage.receipt_hash != spec.lineage_receipt_hash
            or policy.policy_hash != protocol.policy_hash
            or protocol.protocol_hash != spec.public_protocol_hash
            or (
                spec.phase == "confirmation"
                and (
                    development_report.report_hash != spec.development_report_hash
                    or spec.frozen_at < development_report.created_at
                )
            )
        ):
            return False
        regenerated_public, regenerated_private = generate_concept_worldpacks_v312(
            spec, protocol, generated_at=private_pack.generated_at
        )
        if (
            regenerated_public.public_pack_hash != public_pack.public_pack_hash
            or regenerated_private.pack_hash != private_pack.pack_hash
        ):
            return False
        recomputed_bundle = execute_concept_evolution_v312(
            protocol, public_pack, policy, executed_at=bundle.created_at
        )
        if recomputed_bundle.bundle_hash != bundle.bundle_hash:
            return False
        recomputed_report = evaluate_concept_evolution_v312(
            spec, protocol, lineage, private_pack, bundle,
            evaluated_at=report.created_at,
        )
        if recomputed_report.report_hash != report.report_hash:
            return False
        if (
            manifest.phase != spec.phase
            or manifest.terminal_status != report.status
            or [item.model_dump(mode="json") for item in manifest.artifact_refs]
            != [item.model_dump(mode="json") for item in refs[:9]]
        ):
            return False
        event_types = [event["event_type"] for event in events]
        freeze_index = event_types.index(
            "concept_evolution_v312_protocol_frozen_before_private_pack"
        )
        private_index = next(
            index for index, event in enumerate(events)
            if event["event_type"] == "artifact_committed"
            and event["payload"]["kind"] == "private_concept_worldpack_v312"
        )
        return freeze_index < private_index
    except (
        OSError, RuntimeError, ValueError, KeyError, TypeError,
        json.JSONDecodeError, FloatingPointError, AttributeError,
        np.linalg.LinAlgError,
    ):
        return False
