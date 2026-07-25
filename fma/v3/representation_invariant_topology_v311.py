from __future__ import annotations

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
from .skeleton_factorial_v310 import (
    SkeletonFactorialEvolutionReportV310,
    verify_skeleton_factorial_run_v310,
)


DEVELOPMENT_SEEDS_V311 = (26003, 26069, 26141, 26203, 26267, 26321)
CONFIRMATION_SEEDS_V311 = (
    27011, 27077, 27143, 27211, 27277, 27329, 27397, 27457,
)

MechanismV311 = Literal[
    "thermal_relaxation",
    "van_der_pol",
    "lotka_volterra",
    "sir_epidemic",
    "pendulum_open_set",
]
RepresentationV311 = Literal["anonymous_reference", "anonymous_scaled_permuted"]
TopologyV311 = Literal[
    "generic_cubic",
    "scalar_affine_rate",
    "second_order_kinematic",
    "interacting_population",
    "conserved_compartment",
    "uncoupled_linear_decoy",
]

MECHANISMS_V311: tuple[MechanismV311, ...] = (
    "thermal_relaxation",
    "van_der_pol",
    "lotka_volterra",
    "sir_epidemic",
    "pendulum_open_set",
)
REPRESENTATIONS_V311: tuple[RepresentationV311, ...] = (
    "anonymous_reference",
    "anonymous_scaled_permuted",
)
TOPOLOGIES_V311: tuple[TopologyV311, ...] = (
    "generic_cubic",
    "scalar_affine_rate",
    "second_order_kinematic",
    "interacting_population",
    "conserved_compartment",
    "uncoupled_linear_decoy",
)
EXPECTED_TOPOLOGY_V311: dict[MechanismV311, TopologyV311 | None] = {
    "thermal_relaxation": "scalar_affine_rate",
    "van_der_pol": "second_order_kinematic",
    "lotka_volterra": "interacting_population",
    "sir_epidemic": "conserved_compartment",
    "pendulum_open_set": None,
}


def _committed_refs_v311(store: RunStore) -> list[ArtifactRef]:
    events = [
        json.loads(line)
        for line in store.event_path.read_text(encoding="utf-8").splitlines()
    ]
    return [
        ArtifactRef.model_validate(event["payload"])
        for event in events if event["event_type"] == "artifact_committed"
    ]


def _load_one_v311(store: RunStore, refs: list[ArtifactRef], kind: str, model):
    matches = [item for item in refs if item.kind == kind]
    if len(matches) != 1:
        raise ValueError(f"V3.11 requires exactly one {kind}")
    return model.model_validate(store.load_artifact(matches[0]))


def _load_source_v310(run_directory: str | Path) -> SkeletonFactorialEvolutionReportV310:
    source_v391 = (
        Path(run_directory).parents[1]
        / "iteration_17" / "v391_evaluator_partition_recovery"
    )
    if not verify_skeleton_factorial_run_v310(
        run_directory, source_v391_run_directory=source_v391
    ):
        raise ValueError("V3.11 source V3.10 run did not independently verify")
    store = RunStore.open_existing(run_directory)
    refs = _committed_refs_v311(store)
    return _load_one_v311(
        store,
        refs,
        "skeleton_factorial_evolution_report_v310",
        SkeletonFactorialEvolutionReportV310,
    )


class MethodSourceV311(StrictModel):
    source_id: Identifier
    title: Annotated[str, Field(min_length=8)]
    doi: Annotated[str, Field(pattern=r"^10\.")]
    source_url: Annotated[str, Field(pattern=r"^https://")]
    borrowed_principle: Annotated[str, Field(min_length=20)]
    guarantee_transferred: Literal[False] = False


class LiteratureQueryReceiptV311(StrictModel):
    database: Literal["OpenAlex"] = "OpenAlex"
    endpoint: Literal["https://api.openalex.org/works"] = (
        "https://api.openalex.org/works"
    )
    queries: list[str] = Field(min_length=3, max_length=3)
    result_counts: list[Annotated[int, Field(ge=0)]] = Field(min_length=3, max_length=3)
    per_page: Literal[5] = 5
    pages_retrieved_per_query: Literal[1] = 1
    accessed_on: Literal["2026-07-22"] = "2026-07-22"
    targeted_non_exhaustive: Literal[True] = True


class RepresentationMethodEvidenceV311(StrictModel):
    schema_version: Literal["3.11"] = "3.11"
    evidence_id: Identifier
    retrieval: LiteratureQueryReceiptV311
    sources: list[MethodSourceV311] = Field(min_length=3, max_length=3)
    external_content_treated_as_untrusted_data: Literal[True] = True
    evidence_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_evidence(self) -> "RepresentationMethodEvidenceV311":
        if len({item.doi for item in self.sources}) != 3:
            raise ValueError("V3.11 method DOI set differs")
        if self.evidence_hash and self.evidence_hash != self.content_hash():
            raise ValueError("evidence_hash does not match V3.11 evidence")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "evidence_hash")

    def assert_sealed(self) -> None:
        if not self.evidence_hash or self.evidence_hash != self.content_hash():
            raise ValueError("V3.11 method evidence is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "RepresentationMethodEvidenceV311":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"evidence_hash"}),
            evidence_hash=draft.content_hash(),
        )


def default_representation_method_evidence_v311() -> RepresentationMethodEvidenceV311:
    return RepresentationMethodEvidenceV311.seal(
        evidence_id="representation_invariant_topology_method_evidence_v311",
        retrieval=LiteratureQueryReceiptV311(
            queries=[
                "data driven discovery coordinates governing equations dynamical systems",
                "constrained sparse identification nonlinear dynamics conservation laws",
                "dimensional analysis symmetry symbolic regression physical laws",
            ],
            result_counts=[8782, 3518, 3334],
        ),
        sources=[
            MethodSourceV311(
                source_id="champion_etal_2019",
                title="Data-driven discovery of coordinates and governing equations",
                doi="10.1073/pnas.1906995116",
                source_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC6842598/",
                borrowed_principle=(
                    "Treat coordinate choice and governing-equation discovery as coupled hypotheses rather than fixed labels."
                ),
            ),
            MethodSourceV311(
                source_id="loiseau_brunton_2018",
                title="Constrained sparse Galerkin regression",
                doi="10.1017/jfm.2017.823",
                source_url="https://arxiv.org/abs/1611.03271",
                borrowed_principle=(
                    "Encode physical equalities as regression constraints and compare them against unconstrained candidates."
                ),
            ),
            MethodSourceV311(
                source_id="reinbold_etal_2021",
                title="Robust learning from noisy incomplete high-dimensional experimental data via physically constrained symbolic regression",
                doi="10.1038/s41467-021-23479-0",
                source_url="https://www.nature.com/articles/s41467-021-23479-0",
                borrowed_principle=(
                    "Use general physical constraints to narrow a noisy model search while retaining independent predictive tests."
                ),
            ),
        ],
    )


class TopologyDiscoveryPolicyV311(StrictModel):
    schema_version: Literal["3.11"] = "3.11"
    policy_id: Identifier
    method_evidence_hash: Sha256
    source_v310_evolution_hash: Sha256
    topology_catalog: list[TopologyV311] = Field(min_length=6, max_length=6)
    selection_rule: Literal[
        "dual_validation_minimax_parsimony_with_public_loo_switch_guard"
    ] = "dual_validation_minimax_parsimony_with_public_loo_switch_guard"
    state_names_used_as_semantics: Literal[False] = False
    private_mechanism_visible: Literal[False] = False
    private_representation_visible: Literal[False] = False
    private_pair_id_visible: Literal[False] = False
    private_probe_visible: Literal[False] = False
    private_loss_visible: Literal[False] = False
    real_world_execution_permitted: Literal[False] = False
    task_router_permitted: Literal[False] = False
    policy_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_policy(self) -> "TopologyDiscoveryPolicyV311":
        if self.topology_catalog != list(TOPOLOGIES_V311):
            raise ValueError("V3.11 topology catalog differs")
        if self.policy_hash and self.policy_hash != self.content_hash():
            raise ValueError("policy_hash does not match V3.11 policy")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "policy_hash")

    def assert_sealed(self) -> None:
        if not self.policy_hash or self.policy_hash != self.content_hash():
            raise ValueError("V3.11 policy is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "TopologyDiscoveryPolicyV311":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"policy_hash"}),
            policy_hash=draft.content_hash(),
        )


def default_topology_discovery_policy_v311(
    evidence: RepresentationMethodEvidenceV311,
    source_v310_evolution_hash: str,
) -> TopologyDiscoveryPolicyV311:
    evidence.assert_sealed()
    return TopologyDiscoveryPolicyV311.seal(
        policy_id="anonymous_coordinate_topology_discovery_v311",
        method_evidence_hash=evidence.evidence_hash,
        source_v310_evolution_hash=source_v310_evolution_hash,
        topology_catalog=list(TOPOLOGIES_V311),
    )


class PublicTopologyProtocolV311(StrictModel):
    schema_version: Literal["3.11"] = "3.11"
    protocol_id: Identifier
    trajectory_points: Literal[61] = 61
    time_step: Literal[0.03] = 0.03
    public_trajectory_count: Literal[3] = 3
    integral_window_intervals: Literal[6] = 6
    blocked_tail_start_index: Literal[38] = 38
    ridge_alpha: Literal[0.0001] = 0.0001
    sparsity_threshold: Literal[0.015] = 0.015
    maximum_cv_loss: Literal[0.35] = 0.35
    minimum_rank_ratio: Literal[0.9] = 0.9
    maximum_condition_number: Literal[1000000000.0] = 1000000000.0
    unresolved_loss: Literal[10.0] = 10.0
    method_evidence_hash: Sha256
    policy_hash: Sha256
    frozen_at: datetime
    protocol_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_protocol(self) -> "PublicTopologyProtocolV311":
        _assert_timezone(self.frozen_at, "frozen_at")
        if self.protocol_hash and self.protocol_hash != self.content_hash():
            raise ValueError("protocol_hash does not match V3.11 public protocol")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "protocol_hash")

    def assert_sealed(self) -> None:
        if not self.protocol_hash or self.protocol_hash != self.content_hash():
            raise ValueError("V3.11 public protocol is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "PublicTopologyProtocolV311":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"protocol_hash"}),
            protocol_hash=draft.content_hash(),
        )


def default_public_topology_protocol_v311(
    evidence: RepresentationMethodEvidenceV311,
    policy: TopologyDiscoveryPolicyV311,
    *,
    frozen_at: datetime | None = None,
) -> PublicTopologyProtocolV311:
    evidence.assert_sealed()
    policy.assert_sealed()
    return PublicTopologyProtocolV311.seal(
        protocol_id="anonymous_coordinate_topology_public_protocol_v311",
        method_evidence_hash=evidence.evidence_hash,
        policy_hash=policy.policy_hash,
        frozen_at=frozen_at or datetime.now(timezone.utc),
    )


class PrivateTopologyWorldPackSpecV311(StrictModel):
    schema_version: Literal["3.11"] = "3.11"
    experiment_id: Identifier
    phase: Literal["development", "confirmation"]
    mechanisms: list[MechanismV311] = Field(min_length=5, max_length=5)
    representations: list[RepresentationV311] = Field(min_length=2, max_length=2)
    seeds: list[int] = Field(min_length=6, max_length=8)
    observation_noise_fraction: Literal[0.003] = 0.003
    calibration_failure_seed_index: Literal[0] = 0
    expected_quality_case_count: Literal[10] = 10
    public_protocol_hash: Sha256
    source_v310_evolution_hash: Sha256
    development_report_hash: Sha256 | None = None
    bootstrap_replicates: Literal[2000] = 2000
    bootstrap_seed: int
    material_negative_transfer: Literal[0.02] = 0.02
    maximum_negative_transfer_upper_95: Literal[0.1] = 0.1
    minimum_candidate_coverage: Literal[0.9] = 0.9
    minimum_topology_accuracy: Literal[0.8] = 0.8
    minimum_open_set_abstention_rate: Literal[0.9] = 0.9
    minimum_pair_topology_consistency: Literal[0.9] = 0.9
    maximum_pair_loss_difference: Literal[0.03] = 0.03
    maximum_mechanism_regression: Literal[0.02] = 0.02
    maximum_transformed_representation_regression: Literal[0.02] = 0.02
    frozen_delta: Literal[
        "anonymous_state_permutation_unit_scaling_and_unseen_mechanisms_only"
    ] = "anonymous_state_permutation_unit_scaling_and_unseen_mechanisms_only"
    frozen_at: datetime
    spec_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_spec(self) -> "PrivateTopologyWorldPackSpecV311":
        _assert_timezone(self.frozen_at, "frozen_at")
        if self.mechanisms != list(MECHANISMS_V311):
            raise ValueError("V3.11 mechanism order differs")
        if self.representations != list(REPRESENTATIONS_V311):
            raise ValueError("V3.11 representation order differs")
        expected = (
            list(DEVELOPMENT_SEEDS_V311)
            if self.phase == "development"
            else list(CONFIRMATION_SEEDS_V311)
        )
        if self.seeds != expected:
            raise ValueError("V3.11 seeds do not match frozen phase")
        if self.phase == "development" and self.development_report_hash is not None:
            raise ValueError("V3.11 development spec cannot bind itself")
        if self.phase == "confirmation" and self.development_report_hash is None:
            raise ValueError("V3.11 confirmation spec needs development lineage")
        if self.spec_hash and self.spec_hash != self.content_hash():
            raise ValueError("spec_hash does not match V3.11 private spec")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "spec_hash")

    def assert_sealed(self) -> None:
        if not self.spec_hash or self.spec_hash != self.content_hash():
            raise ValueError("V3.11 private spec is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "PrivateTopologyWorldPackSpecV311":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"spec_hash"}),
            spec_hash=draft.content_hash(),
        )


def default_private_topology_spec_v311(
    protocol: PublicTopologyProtocolV311,
    source_v310_evolution_hash: str,
    *,
    phase: Literal["development", "confirmation"],
    development_report_hash: str | None = None,
    frozen_at: datetime | None = None,
) -> PrivateTopologyWorldPackSpecV311:
    protocol.assert_sealed()
    seeds = (
        list(DEVELOPMENT_SEEDS_V311)
        if phase == "development" else list(CONFIRMATION_SEEDS_V311)
    )
    return PrivateTopologyWorldPackSpecV311.seal(
        experiment_id=f"representation_invariant_topology_{phase}_v311",
        phase=phase,
        mechanisms=list(MECHANISMS_V311),
        representations=list(REPRESENTATIONS_V311),
        seeds=seeds,
        public_protocol_hash=protocol.protocol_hash,
        source_v310_evolution_hash=source_v310_evolution_hash,
        development_report_hash=development_report_hash,
        bootstrap_seed=(3110722 if phase == "development" else 3111722),
        frozen_at=frozen_at or datetime.now(timezone.utc),
    )


class AnonymousTrajectoryV311(StrictModel):
    trajectory_id: Identifier
    case_id: Identifier
    times: list[Annotated[float, Field(allow_inf_nan=False)]] = Field(min_length=61, max_length=61)
    states: list[list[Annotated[float, Field(allow_inf_nan=False)]]] = Field(min_length=61, max_length=61)
    trajectory_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_trajectory(self) -> "AnonymousTrajectoryV311":
        if len(self.times) != len(self.states):
            raise ValueError("V3.11 trajectory arrays differ")
        if any(right <= left for left, right in zip(self.times, self.times[1:])):
            raise ValueError("V3.11 times must increase")
        if self.trajectory_hash and self.trajectory_hash != self.content_hash():
            raise ValueError("trajectory_hash does not match V3.11 trajectory")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "trajectory_hash")

    def assert_sealed(self) -> None:
        if not self.trajectory_hash or self.trajectory_hash != self.content_hash():
            raise ValueError("V3.11 trajectory is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "AnonymousTrajectoryV311":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"trajectory_hash"}),
            trajectory_hash=draft.content_hash(),
        )


class PublicTopologyCaseV311(StrictModel):
    schema_version: Literal["3.11"] = "3.11"
    case_id: Identifier
    state_names: list[Identifier] = Field(min_length=1, max_length=3)
    trajectories: list[AnonymousTrajectoryV311] = Field(min_length=3, max_length=3)
    quality_flags: list[Identifier]
    semantic_state_labels_available: Literal[False] = False
    representation_metadata_available: Literal[False] = False
    public_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_case(self) -> "PublicTopologyCaseV311":
        if self.state_names != [f"z{index}" for index in range(len(self.state_names))]:
            raise ValueError("V3.11 public names must be anonymous")
        for trajectory in self.trajectories:
            trajectory.assert_sealed()
            if trajectory.case_id != self.case_id:
                raise ValueError("V3.11 trajectory case differs")
            if any(len(row) != len(self.state_names) for row in trajectory.states):
                raise ValueError("V3.11 state dimension differs")
        if self.public_hash and self.public_hash != self.content_hash():
            raise ValueError("public_hash does not match V3.11 case")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "public_hash")

    def assert_sealed(self) -> None:
        if not self.public_hash or self.public_hash != self.content_hash():
            raise ValueError("V3.11 public case is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "PublicTopologyCaseV311":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"public_hash"}),
            public_hash=draft.content_hash(),
        )


class PublicTopologyWorldPackV311(StrictModel):
    schema_version: Literal["3.11"] = "3.11"
    public_protocol_hash: Sha256
    cases: list[PublicTopologyCaseV311] = Field(min_length=60, max_length=80)
    public_pack_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_pack(self) -> "PublicTopologyWorldPackV311":
        ids = [item.case_id for item in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("V3.11 public case ids differ")
        for case in self.cases:
            case.assert_sealed()
        if self.public_pack_hash and self.public_pack_hash != self.content_hash():
            raise ValueError("public_pack_hash does not match V3.11 public pack")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "public_pack_hash")

    def assert_sealed(self) -> None:
        if not self.public_pack_hash or self.public_pack_hash != self.content_hash():
            raise ValueError("V3.11 public pack is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "PublicTopologyWorldPackV311":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"public_pack_hash"}),
            public_pack_hash=draft.content_hash(),
        )


class PrivateTopologyCaseV311(StrictModel):
    schema_version: Literal["3.11"] = "3.11"
    public_case: PublicTopologyCaseV311
    mechanism: MechanismV311
    representation: RepresentationV311
    hidden_pair_id: Identifier
    hidden_seed: int
    hidden_parameters: dict[str, Annotated[float, Field(allow_inf_nan=False)]]
    observed_to_physical_permutation: list[Annotated[int, Field(ge=0, le=2)]]
    observed_scales: list[Annotated[float, Field(gt=0, allow_inf_nan=False)]]
    private_probe_initials: list[list[Annotated[float, Field(allow_inf_nan=False)]]] = Field(min_length=3, max_length=3)
    private_probe_truths: list[list[list[Annotated[float, Field(allow_inf_nan=False)]]]] = Field(min_length=3, max_length=3)
    private_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_private(self) -> "PrivateTopologyCaseV311":
        self.public_case.assert_sealed()
        dimension = len(self.public_case.state_names)
        if sorted(self.observed_to_physical_permutation) != list(range(dimension)):
            raise ValueError("V3.11 hidden coordinate permutation differs")
        if len(self.observed_scales) != dimension:
            raise ValueError("V3.11 hidden coordinate scales differ")
        if len(self.private_probe_initials) != len(self.private_probe_truths):
            raise ValueError("V3.11 private probe arrays differ")
        if self.private_hash and self.private_hash != self.content_hash():
            raise ValueError("private_hash does not match V3.11 case")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "private_hash")

    def assert_sealed(self) -> None:
        if not self.private_hash or self.private_hash != self.content_hash():
            raise ValueError("V3.11 private case is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "PrivateTopologyCaseV311":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"private_hash"}),
            private_hash=draft.content_hash(),
        )


class PrivateTopologyWorldPackV311(StrictModel):
    schema_version: Literal["3.11"] = "3.11"
    spec_hash: Sha256
    public_pack_hash: Sha256
    cases: list[PrivateTopologyCaseV311] = Field(min_length=60, max_length=80)
    generated_at: datetime
    pack_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_pack(self) -> "PrivateTopologyWorldPackV311":
        _assert_timezone(self.generated_at, "generated_at")
        ids = [item.public_case.case_id for item in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("V3.11 private case ids differ")
        for case in self.cases:
            case.assert_sealed()
        if self.pack_hash and self.pack_hash != self.content_hash():
            raise ValueError("pack_hash does not match V3.11 private pack")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "pack_hash")

    def assert_sealed(self) -> None:
        if not self.pack_hash or self.pack_hash != self.content_hash():
            raise ValueError("V3.11 private pack is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "PrivateTopologyWorldPackV311":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"pack_hash"}),
            pack_hash=draft.content_hash(),
        )


def _mechanism_setup_v311(
    mechanism: MechanismV311,
    random: Random,
) -> tuple[dict[str, float], list[list[float]], list[list[float]]]:
    if mechanism == "thermal_relaxation":
        parameters = {
            "rate": 0.42 + 0.18 * random.random(),
            "equilibrium": 0.65 + 0.35 * random.random(),
        }
        return parameters, [[2.0], [1.55], [2.45]], [[1.25], [2.75], [0.25]]
    if mechanism == "van_der_pol":
        parameters = {"mu": 0.8 + 0.65 * random.random()}
        return (
            parameters,
            [[1.2, 0.0], [-1.0, 0.35], [0.55, -0.85]],
            [[1.75, -0.2], [-1.55, 0.65], [0.15, 1.25]],
        )
    if mechanism == "lotka_volterra":
        parameters = {
            "prey_growth": 0.72 + 0.22 * random.random(),
            "predation": 0.62 + 0.18 * random.random(),
            "predator_decay": 0.78 + 0.22 * random.random(),
            "conversion": 0.55 + 0.2 * random.random(),
        }
        return (
            parameters,
            [[1.2, 0.8], [0.8, 1.3], [1.55, 1.05]],
            [[1.8, 0.55], [0.55, 1.65], [1.45, 1.5]],
        )
    if mechanism == "sir_epidemic":
        parameters = {
            "infection": 1.05 + 0.35 * random.random(),
            "recovery": 0.24 + 0.12 * random.random(),
        }
        return (
            parameters,
            [[0.92, 0.08, 0.0], [0.85, 0.15, 0.0], [0.96, 0.04, 0.0]],
            [[0.78, 0.22, 0.0], [0.97, 0.03, 0.0], [0.88, 0.10, 0.02]],
        )
    parameters = {"damping": 0.08 + 0.08 * random.random()}
    return (
        parameters,
        [[0.8, 0.0], [2.2, 0.0], [-2.4, 0.25]],
        [[2.75, -0.3], [-2.7, 0.55], [1.9, 1.1]],
    )


def _truth_rhs_v311(
    mechanism: MechanismV311,
    state: np.ndarray,
    parameters: dict[str, float],
) -> np.ndarray:
    if mechanism == "thermal_relaxation":
        return np.asarray([
            -parameters["rate"] * (state[0] - parameters["equilibrium"])
        ])
    if mechanism == "van_der_pol":
        return np.asarray([
            state[1],
            parameters["mu"] * (1.0 - state[0] ** 2) * state[1] - state[0],
        ])
    if mechanism == "lotka_volterra":
        prey, predator = state
        return np.asarray([
            parameters["prey_growth"] * prey
            - parameters["predation"] * prey * predator,
            -parameters["predator_decay"] * predator
            + parameters["conversion"] * prey * predator,
        ])
    if mechanism == "sir_epidemic":
        susceptible, infected, _ = state
        infection = parameters["infection"] * susceptible * infected
        recovery = parameters["recovery"] * infected
        return np.asarray([-infection, infection - recovery, recovery])
    return np.asarray([
        state[1],
        -math.sin(state[0]) - parameters["damping"] * state[1],
    ])


def _simulate_truth_v311(
    mechanism: MechanismV311,
    initial: list[float],
    times: list[float],
    parameters: dict[str, float],
) -> np.ndarray:
    solution = solve_ivp(
        lambda _time, state: _truth_rhs_v311(mechanism, state, parameters),
        (float(times[0]), float(times[-1])),
        np.asarray(initial, dtype=float),
        t_eval=np.asarray(times, dtype=float),
        rtol=1e-10,
        atol=1e-12,
        max_step=0.01,
    )
    if not solution.success or not np.isfinite(solution.y).all():
        raise RuntimeError("V3.11 hidden truth simulation failed")
    return solution.y.T


def _representation_v311(
    mechanism: MechanismV311,
    dimension: int,
    seed: int,
    representation: RepresentationV311,
) -> tuple[list[int], list[float]]:
    if representation == "anonymous_reference":
        return list(range(dimension)), [1.0] * dimension
    random = Random(seed * 1000003 + MECHANISMS_V311.index(mechanism) * 10007)
    permutation = list(range(dimension))
    random.shuffle(permutation)
    if dimension > 1 and permutation == list(range(dimension)):
        permutation = permutation[1:] + permutation[:1]
    if mechanism == "sir_epidemic":
        common = math.exp(random.uniform(math.log(0.45), math.log(2.4)))
        scales = [common] * dimension
    else:
        scales = [
            math.exp(random.uniform(math.log(0.45), math.log(2.4)))
            for _ in range(dimension)
        ]
    return permutation, scales


def _to_observed_v311(
    physical: np.ndarray,
    permutation: list[int],
    scales: list[float],
) -> np.ndarray:
    values = physical[:, permutation] * np.asarray(scales, dtype=float)[None, :]
    if not np.isfinite(values).all():
        raise RuntimeError("V3.11 coordinate transform produced nonfinite values")
    return values


def _noisy_v311(
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
        noisy = noisy + 0.25 * scale[np.newaxis, :]
    return noisy


def generate_topology_worldpacks_v311(
    private_spec: PrivateTopologyWorldPackSpecV311,
    public_protocol: PublicTopologyProtocolV311,
    *,
    generated_at: datetime | None = None,
) -> tuple[PublicTopologyWorldPackV311, PrivateTopologyWorldPackV311]:
    private_spec.assert_sealed()
    public_protocol.assert_sealed()
    if private_spec.public_protocol_hash != public_protocol.protocol_hash:
        raise ValueError("V3.11 generation protocol binding differs")
    times = [
        index * public_protocol.time_step
        for index in range(public_protocol.trajectory_points)
    ]
    private_cases: list[PrivateTopologyCaseV311] = []
    for seed_index, seed in enumerate(private_spec.seeds):
        for mechanism_index, mechanism in enumerate(private_spec.mechanisms):
            random = Random(seed * 104729 + mechanism_index * 7919)
            parameters, public_initials, private_initials = _mechanism_setup_v311(
                mechanism, random
            )
            public_physical = [
                _simulate_truth_v311(mechanism, initial, times, parameters)
                for initial in public_initials
            ]
            private_physical = [
                _simulate_truth_v311(mechanism, initial, times, parameters)
                for initial in private_initials
            ]
            dimension = len(public_initials[0])
            hidden_pair_id = f"pair_{sha256_value([seed, mechanism])[:16]}"
            calibration_failed = (
                seed_index == private_spec.calibration_failure_seed_index
            )
            for representation_index, representation in enumerate(
                private_spec.representations
            ):
                permutation, scales = _representation_v311(
                    mechanism, dimension, seed, representation
                )
                case_id = f"case_{sha256_value([seed, mechanism, representation])[:16]}"
                trajectories = []
                for trajectory_index, physical in enumerate(public_physical):
                    observed = _to_observed_v311(physical, permutation, scales)
                    observed = _noisy_v311(
                        observed,
                        private_spec.observation_noise_fraction,
                        seed=(
                            seed * 2000003 + mechanism_index * 20011
                            + representation_index * 1009 + trajectory_index * 101
                        ),
                        calibration_failed=calibration_failed,
                    )
                    trajectories.append(AnonymousTrajectoryV311.seal(
                        trajectory_id=f"trajectory_{case_id}_{trajectory_index}",
                        case_id=case_id,
                        times=times,
                        states=observed.tolist(),
                    ))
                public_case = PublicTopologyCaseV311.seal(
                    case_id=case_id,
                    state_names=[f"z{index}" for index in range(dimension)],
                    trajectories=trajectories,
                    quality_flags=(
                        ["sensor_calibration_failed"] if calibration_failed else []
                    ),
                )
                private_initial_observed = [
                    _to_observed_v311(
                        np.asarray(initial, dtype=float)[None, :],
                        permutation,
                        scales,
                    )[0].tolist()
                    for initial in private_initials
                ]
                private_truth_observed = [
                    _to_observed_v311(physical, permutation, scales).tolist()
                    for physical in private_physical
                ]
                private_cases.append(PrivateTopologyCaseV311.seal(
                    public_case=public_case,
                    mechanism=mechanism,
                    representation=representation,
                    hidden_pair_id=hidden_pair_id,
                    hidden_seed=seed,
                    hidden_parameters=parameters,
                    observed_to_physical_permutation=permutation,
                    observed_scales=scales,
                    private_probe_initials=private_initial_observed,
                    private_probe_truths=private_truth_observed,
                ))
    public_pack = PublicTopologyWorldPackV311.seal(
        public_protocol_hash=public_protocol.protocol_hash,
        cases=[item.public_case for item in private_cases],
    )
    private_pack = PrivateTopologyWorldPackV311.seal(
        spec_hash=private_spec.spec_hash,
        public_pack_hash=public_pack.public_pack_hash,
        cases=private_cases,
        generated_at=generated_at or datetime.now(timezone.utc),
    )
    return public_pack, private_pack


class TopologyModelV311(StrictModel):
    schema_version: Literal["3.11"] = "3.11"
    model_id: Identifier
    case_id: Identifier
    topology: TopologyV311
    role_mapping: list[Annotated[int, Field(ge=0, le=2)]]
    state_names: list[Identifier] = Field(min_length=1, max_length=3)
    basis_terms: list[PolynomialBasisTermV24]
    coefficient_matrix: list[list[Annotated[float, Field(allow_inf_nan=False)]]]
    source_trajectory_hashes: list[Sha256]
    normalized_rank_ratio: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    normalized_condition_number: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    normalized_integral_residual: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    structural_identifiability_proven: Literal[False] = False
    model_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_model(self) -> "TopologyModelV311":
        if len(self.coefficient_matrix) != len(self.state_names):
            raise ValueError("V3.11 model equation count differs")
        if any(len(row) != len(self.basis_terms) for row in self.coefficient_matrix):
            raise ValueError("V3.11 model coefficient width differs")
        if any(len(term.exponents) != len(self.state_names) for term in self.basis_terms):
            raise ValueError("V3.11 model term dimension differs")
        if self.model_hash and self.model_hash != self.content_hash():
            raise ValueError("model_hash does not match V3.11 topology model")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "model_hash")

    def assert_sealed(self) -> None:
        if not self.model_hash or self.model_hash != self.content_hash():
            raise ValueError("V3.11 topology model is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "TopologyModelV311":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"model_hash"}),
            model_hash=draft.content_hash(),
        )


@dataclass(frozen=True)
class _CandidateDefinitionV311:
    candidate_id: str
    topology: TopologyV311
    role_mapping: list[int]


@dataclass(frozen=True)
class _FitOutcomeV311:
    model: TopologyModelV311
    rank_ratio: float
    condition_number: float
    residual: float


def _term_v311(term_id: str, exponents: list[int]) -> PolynomialBasisTermV24:
    return PolynomialBasisTermV24(term_id=term_id, exponents=exponents)


def _candidate_definitions_v311(dimension: int) -> list[_CandidateDefinitionV311]:
    definitions = [_CandidateDefinitionV311("generic_cubic", "generic_cubic", [])]
    if dimension == 1:
        definitions.append(_CandidateDefinitionV311(
            "scalar_affine_rate", "scalar_affine_rate", [0]
        ))
    elif dimension == 2:
        definitions.extend([
            _CandidateDefinitionV311(
                "oscillator_position0_velocity1",
                "second_order_kinematic",
                [0, 1],
            ),
            _CandidateDefinitionV311(
                "oscillator_position1_velocity0",
                "second_order_kinematic",
                [1, 0],
            ),
            _CandidateDefinitionV311(
                "interacting_population", "interacting_population", [0, 1]
            ),
            _CandidateDefinitionV311(
                "uncoupled_linear_decoy", "uncoupled_linear_decoy", [0, 1]
            ),
        ])
    else:
        definitions.extend([
            _CandidateDefinitionV311(
                f"compartment_s{roles[0]}_i{roles[1]}_r{roles[2]}",
                "conserved_compartment",
                list(roles),
            )
            for roles in itertools.permutations(range(3))
        ])
        definitions.append(_CandidateDefinitionV311(
            "uncoupled_linear_decoy", "uncoupled_linear_decoy", [0, 1, 2]
        ))
    return definitions


def _terms_and_masks_v311(
    state_names: list[str],
    definition: _CandidateDefinitionV311,
) -> tuple[list[PolynomialBasisTermV24], list[list[int]] | None]:
    dimension = len(state_names)
    if definition.topology == "generic_cubic":
        terms = polynomial_basis_terms(state_names, 3)
        return terms, [list(range(len(terms))) for _ in range(dimension)]
    if definition.topology == "scalar_affine_rate":
        return [
            _term_v311("one", [0]),
            _term_v311(state_names[0], [1]),
        ], [[0, 1]]
    if definition.topology == "second_order_kinematic":
        position, velocity = definition.role_mapping
        position_exp = [0] * dimension
        position_exp[position] = 1
        velocity_exp = [0] * dimension
        velocity_exp[velocity] = 1
        nonlinear_exp = [0] * dimension
        nonlinear_exp[position] = 2
        nonlinear_exp[velocity] = 1
        terms = [
            _term_v311(state_names[position], position_exp),
            _term_v311(state_names[velocity], velocity_exp),
            _term_v311(
                f"{state_names[position]}2_{state_names[velocity]}", nonlinear_exp
            ),
        ]
        masks = [[] for _ in range(dimension)]
        masks[position] = [1]
        masks[velocity] = [0, 1, 2]
        return terms, masks
    if definition.topology == "interacting_population":
        interaction = [1, 1]
        terms = [
            _term_v311(state_names[0], [1, 0]),
            _term_v311(state_names[1], [0, 1]),
            _term_v311(f"{state_names[0]}_{state_names[1]}", interaction),
        ]
        return terms, [[0, 2], [1, 2]]
    if definition.topology == "conserved_compartment":
        susceptible, infected, _ = definition.role_mapping
        infection_exp = [0, 0, 0]
        infection_exp[susceptible] = 1
        infection_exp[infected] = 1
        infected_exp = [0, 0, 0]
        infected_exp[infected] = 1
        return [
            _term_v311(
                f"{state_names[susceptible]}_{state_names[infected]}",
                infection_exp,
            ),
            _term_v311(state_names[infected], infected_exp),
        ], None
    terms = [
        _term_v311(
            state_names[index],
            [int(other == index) for other in range(dimension)],
        )
        for index in range(dimension)
    ]
    return terms, [[index] for index in range(dimension)]


def _integral_arrays_v311(
    trajectories: list[AnonymousTrajectoryV311],
    terms: list[PolynomialBasisTermV24],
    protocol: PublicTopologyProtocolV311,
) -> tuple[np.ndarray, np.ndarray]:
    libraries: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    window = protocol.integral_window_intervals
    for trajectory in trajectories:
        states = np.asarray(trajectory.states, dtype=float)
        times = np.asarray(trajectory.times, dtype=float)
        library = evaluate_polynomial_library(states, terms)
        for start in range(len(states) - window):
            end = start + window
            libraries.append(np.trapezoid(
                library[start:end + 1], times[start:end + 1], axis=0
            )[None, :])
            targets.append((states[end] - states[start])[None, :])
    return np.vstack(libraries), np.vstack(targets)


def _ridge_v311(
    library: np.ndarray,
    targets: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, int, float]:
    scales = np.sqrt(np.mean(library**2, axis=0))
    scales[scales < 1e-12] = 1.0
    normalized = library / scales
    solved = np.linalg.solve(
        normalized.T @ normalized + alpha * np.eye(normalized.shape[1]),
        normalized.T @ targets,
    ).T / scales[np.newaxis, :]
    rank = int(np.linalg.matrix_rank(normalized, tol=1e-10))
    condition = float(np.linalg.cond(normalized))
    if not math.isfinite(condition):
        condition = 1e15
    return solved, rank, condition


def _fit_masked_v311(
    library: np.ndarray,
    targets: np.ndarray,
    masks: list[list[int]],
    protocol: PublicTopologyProtocolV311,
    *,
    sparse: bool,
) -> tuple[np.ndarray, float, float]:
    coefficients = np.zeros((targets.shape[1], library.shape[1]), dtype=float)
    rank_ratios: list[float] = []
    conditions: list[float] = []
    for equation, allowed in enumerate(masks):
        if not allowed:
            continue
        selected = library[:, allowed]
        solved, rank, condition = _ridge_v311(
            selected, targets[:, [equation]], protocol.ridge_alpha
        )
        row = solved[0]
        if sparse:
            active = np.abs(row) >= protocol.sparsity_threshold
            for _ in range(12):
                previous = active.copy()
                row[~active] = 0.0
                if active.any():
                    updated, _, _ = _ridge_v311(
                        selected[:, active],
                        targets[:, [equation]],
                        protocol.ridge_alpha,
                    )
                    row[active] = updated[0]
                active = np.abs(row) >= protocol.sparsity_threshold
                if np.array_equal(active, previous):
                    break
            row[~active] = 0.0
        coefficients[equation, allowed] = row
        rank_ratios.append(rank / max(len(allowed), 1))
        conditions.append(condition)
    return (
        coefficients,
        min(rank_ratios) if rank_ratios else 0.0,
        max(conditions) if conditions else 1e15,
    )


def _fit_compartment_v311(
    library: np.ndarray,
    targets: np.ndarray,
    roles: list[int],
    protocol: PublicTopologyProtocolV311,
) -> tuple[np.ndarray, float, float]:
    susceptible, infected, recovered = roles
    rows: list[list[float]] = []
    values: list[float] = []
    for index in range(len(library)):
        infection_feature, recovery_feature = library[index]
        for equation in range(3):
            if equation == susceptible:
                rows.append([-infection_feature, 0.0])
            elif equation == infected:
                rows.append([infection_feature, -recovery_feature])
            elif equation == recovered:
                rows.append([0.0, recovery_feature])
            values.append(targets[index, equation])
    design = np.asarray(rows, dtype=float)
    response = np.asarray(values, dtype=float)[:, None]
    solved, rank, condition = _ridge_v311(
        design, response, protocol.ridge_alpha
    )
    infection, recovery = solved[0]
    coefficients = np.zeros((3, 2), dtype=float)
    coefficients[susceptible] = [-infection, 0.0]
    coefficients[infected] = [infection, -recovery]
    coefficients[recovered] = [0.0, recovery]
    return coefficients, rank / 2.0, condition


def _fit_model_v311(
    case: PublicTopologyCaseV311,
    trajectories: list[AnonymousTrajectoryV311],
    definition: _CandidateDefinitionV311,
    protocol: PublicTopologyProtocolV311,
    *,
    suffix: str,
) -> _FitOutcomeV311:
    terms, masks = _terms_and_masks_v311(case.state_names, definition)
    library, targets = _integral_arrays_v311(trajectories, terms, protocol)
    if definition.topology == "conserved_compartment":
        coefficients, rank_ratio, condition = _fit_compartment_v311(
            library, targets, definition.role_mapping, protocol
        )
    else:
        coefficients, rank_ratio, condition = _fit_masked_v311(
            library,
            targets,
            masks,
            protocol,
            sparse=(definition.topology == "generic_cubic"),
        )
    fitted = library @ coefficients.T
    residual = float(
        np.sqrt(np.mean((fitted - targets) ** 2))
        / max(float(np.sqrt(np.mean(targets**2))), 0.05)
    )
    model = TopologyModelV311.seal(
        model_id=f"model_{case.case_id}_{definition.candidate_id}_{suffix}",
        case_id=case.case_id,
        topology=definition.topology,
        role_mapping=definition.role_mapping,
        state_names=case.state_names,
        basis_terms=terms,
        coefficient_matrix=coefficients.tolist(),
        source_trajectory_hashes=[item.trajectory_hash for item in trajectories],
        normalized_rank_ratio=rank_ratio,
        normalized_condition_number=condition,
        normalized_integral_residual=residual,
    )
    return _FitOutcomeV311(model, rank_ratio, condition, residual)


def _simulate_model_v311(
    model: TopologyModelV311,
    initial: list[float],
    times: list[float],
) -> list[list[float]]:
    model.assert_sealed()
    coefficients = np.asarray(model.coefficient_matrix, dtype=float)

    def rhs(_time: float, state: np.ndarray) -> np.ndarray:
        if np.max(np.abs(state)) > 1e6:
            raise FloatingPointError("V3.11 fitted topology diverged")
        library = evaluate_polynomial_library(
            state.reshape(1, -1), model.basis_terms
        )[0]
        return coefficients @ library

    try:
        solution = solve_ivp(
            rhs,
            (float(times[0]), float(times[-1])),
            np.asarray(initial, dtype=float),
            t_eval=np.asarray(times, dtype=float),
            rtol=1e-8,
            atol=1e-10,
            max_step=0.01,
        )
    except (FloatingPointError, ValueError) as exc:
        raise RuntimeError("V3.11 fitted topology simulation failed") from exc
    if not solution.success or not np.isfinite(solution.y).all():
        raise RuntimeError("V3.11 fitted topology did not cover trajectory")
    return solution.y.T.tolist()


@dataclass(frozen=True)
class _TrajectoryViewV311:
    trajectory_id: str
    case_id: str
    times: list[float]
    states: list[list[float]]
    trajectory_hash: str


def _prefix_v311(
    trajectory: AnonymousTrajectoryV311,
    end: int,
) -> _TrajectoryViewV311:
    content = {
        "source_trajectory_hash": trajectory.trajectory_hash,
        "exclusive_end": end,
        "times": trajectory.times[:end],
        "states": trajectory.states[:end],
    }
    return _TrajectoryViewV311(
        trajectory_id=f"prefix_{trajectory.trajectory_id}_{end}",
        case_id=trajectory.case_id,
        times=trajectory.times[:end],
        states=trajectory.states[:end],
        trajectory_hash=sha256_value(content),
    )


class TopologyChallengeReceiptV311(StrictModel):
    schema_version: Literal["3.11"] = "3.11"
    challenge_id: Identifier
    case_id: Identifier
    candidate_id: Identifier
    topology: TopologyV311
    role_mapping: list[Annotated[int, Field(ge=0, le=2)]]
    loo_losses: list[Annotated[float, Field(ge=0, allow_inf_nan=False)]] = Field(min_length=3, max_length=3)
    blocked_tail_losses: list[Annotated[float, Field(ge=0, allow_inf_nan=False)]] = Field(min_length=3, max_length=3)
    loo_mean: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    loo_standard_error: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    blocked_tail_mean: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    robust_validation_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    simulation_failure_count: Annotated[int, Field(ge=0, le=6)]
    final_model: TopologyModelV311
    eligible: bool
    semantic_state_labels_used: Literal[False] = False
    private_values_used: Literal[False] = False
    challenge_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_challenge(self) -> "TopologyChallengeReceiptV311":
        self.final_model.assert_sealed()
        loo = np.asarray(self.loo_losses, dtype=float)
        blocked = np.asarray(self.blocked_tail_losses, dtype=float)
        if not math.isclose(self.loo_mean, float(np.mean(loo)), abs_tol=1e-12):
            raise ValueError("V3.11 LOO mean does not recompute")
        if not math.isclose(
            self.loo_standard_error,
            float(np.std(loo, ddof=1) / math.sqrt(3)),
            abs_tol=1e-12,
        ):
            raise ValueError("V3.11 LOO SE does not recompute")
        if not math.isclose(
            self.blocked_tail_mean, float(np.mean(blocked)), abs_tol=1e-12
        ):
            raise ValueError("V3.11 blocked mean does not recompute")
        if not math.isclose(
            self.robust_validation_loss,
            max(self.loo_mean, self.blocked_tail_mean),
            abs_tol=1e-12,
        ):
            raise ValueError("V3.11 robust loss does not recompute")
        if self.challenge_hash and self.challenge_hash != self.content_hash():
            raise ValueError("challenge_hash does not match V3.11 challenge")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "challenge_hash")

    def assert_sealed(self) -> None:
        if not self.challenge_hash or self.challenge_hash != self.content_hash():
            raise ValueError("V3.11 challenge is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "TopologyChallengeReceiptV311":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"challenge_hash"}),
            challenge_hash=draft.content_hash(),
        )


def _challenge_v311(
    case: PublicTopologyCaseV311,
    definition: _CandidateDefinitionV311,
    protocol: PublicTopologyProtocolV311,
) -> TopologyChallengeReceiptV311:
    trajectories = case.trajectories
    loo_losses: list[float] = []
    blocked_losses: list[float] = []
    failures = 0
    for holdout_index, holdout in enumerate(trajectories):
        training = [
            item for index, item in enumerate(trajectories)
            if index != holdout_index
        ]
        fit = _fit_model_v311(
            case, training, definition, protocol, suffix=f"loo{holdout_index}"
        )
        try:
            prediction = _simulate_model_v311(
                fit.model, holdout.states[0], holdout.times
            )
            loo_losses.append(trajectory_nrmse(holdout.states, prediction))
        except RuntimeError:
            loo_losses.append(protocol.unresolved_loss)
            failures += 1
    prefix_end = protocol.blocked_tail_start_index + 1
    prefixes = [_prefix_v311(item, prefix_end) for item in trajectories]
    blocked_fit = _fit_model_v311(
        case, prefixes, definition, protocol, suffix="blocked_prefixes"
    )
    for holdout in trajectories:
        tail_states = holdout.states[protocol.blocked_tail_start_index:]
        tail_times = holdout.times[protocol.blocked_tail_start_index:]
        try:
            prediction = _simulate_model_v311(
                blocked_fit.model, tail_states[0], tail_times
            )
            blocked_losses.append(trajectory_nrmse(tail_states, prediction))
        except RuntimeError:
            blocked_losses.append(protocol.unresolved_loss)
            failures += 1
    final_fit = _fit_model_v311(
        case, trajectories, definition, protocol, suffix="all_public"
    )
    loo_values = np.asarray(loo_losses, dtype=float)
    blocked_values = np.asarray(blocked_losses, dtype=float)
    loo_mean = float(np.mean(loo_values))
    blocked_mean = float(np.mean(blocked_values))
    eligible = (
        failures == 0
        and max(loo_mean, blocked_mean) <= protocol.maximum_cv_loss
        and final_fit.rank_ratio >= protocol.minimum_rank_ratio
        and final_fit.condition_number <= protocol.maximum_condition_number
    )
    return TopologyChallengeReceiptV311.seal(
        challenge_id=f"challenge_{case.case_id}_{definition.candidate_id}",
        case_id=case.case_id,
        candidate_id=definition.candidate_id,
        topology=definition.topology,
        role_mapping=definition.role_mapping,
        loo_losses=loo_losses,
        blocked_tail_losses=blocked_losses,
        loo_mean=loo_mean,
        loo_standard_error=float(
            np.std(loo_values, ddof=1) / math.sqrt(len(loo_values))
        ),
        blocked_tail_mean=blocked_mean,
        robust_validation_loss=max(loo_mean, blocked_mean),
        simulation_failure_count=failures,
        final_model=final_fit.model,
        eligible=eligible,
    )


class TopologyDecisionV311(StrictModel):
    schema_version: Literal["3.11"] = "3.11"
    decision_id: Identifier
    case_id: Identifier
    arm: Literal["generic_cubic_baseline", "topology_discovery_candidate"]
    challenge_hashes: list[Sha256]
    decision: Literal["select", "abstain"]
    reason: Literal[
        "public_quality_failure",
        "no_eligible_topology",
        "public_dual_validation_selection",
    ]
    topology_hypothesis_challenge_hash: Sha256 | None
    topology_hypothesis: TopologyV311 | None
    topology_hypothesis_role_mapping: list[
        Annotated[int, Field(ge=0, le=2)]
    ] | None
    selected_challenge_hash: Sha256 | None
    selected_topology: TopologyV311 | None
    selected_role_mapping: list[Annotated[int, Field(ge=0, le=2)]] | None
    selected_model_hash: Sha256 | None
    semantic_state_labels_used: Literal[False] = False
    private_values_used: Literal[False] = False
    decision_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_decision(self) -> "TopologyDecisionV311":
        selected = self.decision == "select"
        prediction_fields = (
            self.selected_challenge_hash,
            self.selected_topology,
            self.selected_role_mapping,
            self.selected_model_hash,
        )
        hypothesis_fields = (
            self.topology_hypothesis_challenge_hash,
            self.topology_hypothesis,
            self.topology_hypothesis_role_mapping,
        )
        if selected != all(item is not None for item in prediction_fields):
            raise ValueError("V3.11 decision selection fields differ")
        if selected != all(item is not None for item in hypothesis_fields):
            raise ValueError("V3.11 decision hypothesis fields differ")
        if self.decision_hash and self.decision_hash != self.content_hash():
            raise ValueError("decision_hash does not match V3.11 decision")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "decision_hash")

    def assert_sealed(self) -> None:
        if not self.decision_hash or self.decision_hash != self.content_hash():
            raise ValueError("V3.11 decision is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "TopologyDecisionV311":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"decision_hash"}),
            decision_hash=draft.content_hash(),
        )


def _abstain_v311(
    case_id: str,
    arm: Literal["generic_cubic_baseline", "topology_discovery_candidate"],
    reason: Literal["public_quality_failure", "no_eligible_topology"],
    challenge_hashes: list[str],
) -> TopologyDecisionV311:
    return TopologyDecisionV311.seal(
        decision_id=f"decision_{case_id}_{arm}",
        case_id=case_id,
        arm=arm,
        challenge_hashes=challenge_hashes,
        decision="abstain",
        reason=reason,
        topology_hypothesis_challenge_hash=None,
        topology_hypothesis=None,
        topology_hypothesis_role_mapping=None,
        selected_challenge_hash=None,
        selected_topology=None,
        selected_role_mapping=None,
        selected_model_hash=None,
    )


def _selected_decision_v311(
    case_id: str,
    arm: Literal["generic_cubic_baseline", "topology_discovery_candidate"],
    selected: TopologyChallengeReceiptV311,
    challenge_hashes: list[str],
    *,
    hypothesis: TopologyChallengeReceiptV311 | None = None,
) -> TopologyDecisionV311:
    hypothesis = hypothesis or selected
    return TopologyDecisionV311.seal(
        decision_id=f"decision_{case_id}_{arm}",
        case_id=case_id,
        arm=arm,
        challenge_hashes=challenge_hashes,
        decision="select",
        reason="public_dual_validation_selection",
        topology_hypothesis_challenge_hash=hypothesis.challenge_hash,
        topology_hypothesis=hypothesis.topology,
        topology_hypothesis_role_mapping=hypothesis.role_mapping,
        selected_challenge_hash=selected.challenge_hash,
        selected_topology=selected.topology,
        selected_role_mapping=selected.role_mapping,
        selected_model_hash=selected.final_model.model_hash,
    )


def _select_v311(
    case: PublicTopologyCaseV311,
    challenges: list[TopologyChallengeReceiptV311],
) -> tuple[TopologyDecisionV311, TopologyDecisionV311]:
    generic = next(item for item in challenges if item.topology == "generic_cubic")
    if generic.eligible:
        baseline = _selected_decision_v311(
            case.case_id,
            "generic_cubic_baseline",
            generic,
            [generic.challenge_hash],
        )
    else:
        baseline = _abstain_v311(
            case.case_id,
            "generic_cubic_baseline",
            "no_eligible_topology",
            [generic.challenge_hash],
        )
    eligible = [item for item in challenges if item.eligible]
    if not eligible:
        candidate = _abstain_v311(
            case.case_id,
            "topology_discovery_candidate",
            "no_eligible_topology",
            [],
        )
        return baseline, candidate
    best = min(eligible, key=lambda item: item.robust_validation_loss)
    near_best = [
        item for item in eligible
        if item.robust_validation_loss
        <= best.robust_validation_loss + best.loo_standard_error
    ]
    parsimony_order = {
        "scalar_affine_rate": 0,
        "second_order_kinematic": 0,
        "interacting_population": 0,
        "conserved_compartment": 0,
        "generic_cubic": 2,
        "uncoupled_linear_decoy": 3,
    }
    selected = min(
        near_best,
        key=lambda item: (
            parsimony_order[item.topology],
            item.robust_validation_loss,
            item.candidate_id,
        ),
    )
    hypothesis = selected
    if (
        generic.eligible
        and selected.topology != "generic_cubic"
        and selected.loo_mean + selected.loo_standard_error >= generic.loo_mean
    ):
        selected = generic
    candidate = _selected_decision_v311(
        case.case_id,
        "topology_discovery_candidate",
        selected,
        [item.challenge_hash for item in eligible],
        hypothesis=hypothesis,
    )
    return baseline, candidate


class TopologyCaseReceiptV311(StrictModel):
    schema_version: Literal["3.11"] = "3.11"
    receipt_id: Identifier
    case_id: Identifier
    public_case_hash: Sha256
    policy_hash: Sha256
    quality_flags: list[Identifier]
    challenges: list[TopologyChallengeReceiptV311]
    baseline_decision: TopologyDecisionV311
    candidate_decision: TopologyDecisionV311
    executed_at: datetime
    receipt_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_receipt(self) -> "TopologyCaseReceiptV311":
        _assert_timezone(self.executed_at, "executed_at")
        self.baseline_decision.assert_sealed()
        self.candidate_decision.assert_sealed()
        for challenge in self.challenges:
            challenge.assert_sealed()
            if challenge.case_id != self.case_id:
                raise ValueError("V3.11 challenge case differs")
        hashes = {item.challenge_hash for item in self.challenges}
        for decision in (self.baseline_decision, self.candidate_decision):
            if not set(decision.challenge_hashes).issubset(hashes):
                raise ValueError("V3.11 decision references unknown challenge")
            if decision.selected_challenge_hash and decision.selected_challenge_hash not in hashes:
                raise ValueError("V3.11 selected challenge missing")
            if (
                decision.topology_hypothesis_challenge_hash
                and decision.topology_hypothesis_challenge_hash not in hashes
            ):
                raise ValueError("V3.11 topology hypothesis challenge missing")
        if self.quality_flags:
            if self.challenges:
                raise ValueError("V3.11 quality case cannot execute discovery")
            if (
                self.baseline_decision.reason != "public_quality_failure"
                or self.candidate_decision.reason != "public_quality_failure"
            ):
                raise ValueError("V3.11 quality abstention differs")
        else:
            expected_count = {1: 2, 2: 5, 3: 8}[
                len(self.challenges[0].final_model.state_names)
            ]
            if len(self.challenges) != expected_count:
                raise ValueError("V3.11 public candidate matrix incomplete")
        if self.receipt_hash and self.receipt_hash != self.content_hash():
            raise ValueError("receipt_hash does not match V3.11 case receipt")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "receipt_hash")

    def assert_sealed(self) -> None:
        if not self.receipt_hash or self.receipt_hash != self.content_hash():
            raise ValueError("V3.11 case receipt is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "TopologyCaseReceiptV311":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"receipt_hash"}),
            receipt_hash=draft.content_hash(),
        )


class TopologyDiscoveryBundleV311(StrictModel):
    schema_version: Literal["3.11"] = "3.11"
    bundle_id: Identifier
    public_protocol_hash: Sha256
    public_pack_hash: Sha256
    policy_hash: Sha256
    case_receipts: list[TopologyCaseReceiptV311] = Field(min_length=60, max_length=80)
    created_at: datetime
    bundle_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_bundle(self) -> "TopologyDiscoveryBundleV311":
        _assert_timezone(self.created_at, "created_at")
        ids = [item.case_id for item in self.case_receipts]
        if len(ids) != len(set(ids)):
            raise ValueError("V3.11 bundle case ids differ")
        for receipt in self.case_receipts:
            receipt.assert_sealed()
            if receipt.policy_hash != self.policy_hash:
                raise ValueError("V3.11 bundle policy binding differs")
        if self.bundle_hash and self.bundle_hash != self.content_hash():
            raise ValueError("bundle_hash does not match V3.11 bundle")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "bundle_hash")

    def assert_sealed(self) -> None:
        if not self.bundle_hash or self.bundle_hash != self.content_hash():
            raise ValueError("V3.11 bundle is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "TopologyDiscoveryBundleV311":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"bundle_hash"}),
            bundle_hash=draft.content_hash(),
        )


def execute_topology_discovery_v311(
    public_protocol: PublicTopologyProtocolV311,
    public_pack: PublicTopologyWorldPackV311,
    policy: TopologyDiscoveryPolicyV311,
    *,
    executed_at: datetime,
) -> TopologyDiscoveryBundleV311:
    for artifact in (public_protocol, public_pack, policy):
        artifact.assert_sealed()
    if (
        public_pack.public_protocol_hash != public_protocol.protocol_hash
        or policy.policy_hash != public_protocol.policy_hash
    ):
        raise ValueError("V3.11 public execution binding differs")
    receipts: list[TopologyCaseReceiptV311] = []
    for case in public_pack.cases:
        if case.quality_flags:
            baseline = _abstain_v311(
                case.case_id,
                "generic_cubic_baseline",
                "public_quality_failure",
                [],
            )
            candidate = _abstain_v311(
                case.case_id,
                "topology_discovery_candidate",
                "public_quality_failure",
                [],
            )
            challenges: list[TopologyChallengeReceiptV311] = []
        else:
            challenges = [
                _challenge_v311(case, definition, public_protocol)
                for definition in _candidate_definitions_v311(len(case.state_names))
            ]
            baseline, candidate = _select_v311(case, challenges)
        receipts.append(TopologyCaseReceiptV311.seal(
            receipt_id=f"receipt_{case.case_id}",
            case_id=case.case_id,
            public_case_hash=case.public_hash,
            policy_hash=policy.policy_hash,
            quality_flags=case.quality_flags,
            challenges=challenges,
            baseline_decision=baseline,
            candidate_decision=candidate,
            executed_at=executed_at,
        ))
    return TopologyDiscoveryBundleV311.seal(
        bundle_id=f"bundle_{public_protocol.protocol_id}",
        public_protocol_hash=public_protocol.protocol_hash,
        public_pack_hash=public_pack.public_pack_hash,
        policy_hash=policy.policy_hash,
        case_receipts=receipts,
        created_at=executed_at,
    )


class PrivateTopologyCaseResultV311(StrictModel):
    schema_version: Literal["3.11"] = "3.11"
    case_id: Identifier
    hidden_pair_id: Identifier
    mechanism: MechanismV311
    representation: RepresentationV311
    baseline_topology: TopologyV311 | None
    candidate_topology: TopologyV311 | None
    baseline_target_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    candidate_target_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    candidate_improvement: Annotated[float, Field(allow_inf_nan=False)]
    topology_correct: bool
    material_negative_transfer: bool
    private_values_visible_to_generator: Literal[False] = False

    @model_validator(mode="after")
    def validate_result(self) -> "PrivateTopologyCaseResultV311":
        if not math.isclose(
            self.candidate_improvement,
            self.baseline_target_loss - self.candidate_target_loss,
            abs_tol=1e-12,
        ):
            raise ValueError("V3.11 case improvement does not recompute")
        return self


class RepresentationPairResultV311(StrictModel):
    schema_version: Literal["3.11"] = "3.11"
    hidden_pair_id: Identifier
    mechanism: MechanismV311
    reference_case_id: Identifier
    transformed_case_id: Identifier
    reference_topology: TopologyV311 | None
    transformed_topology: TopologyV311 | None
    topology_consistent: bool
    reference_target_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    transformed_target_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    absolute_target_loss_difference: Annotated[float, Field(ge=0, allow_inf_nan=False)]

    @model_validator(mode="after")
    def validate_pair(self) -> "RepresentationPairResultV311":
        if not math.isclose(
            self.absolute_target_loss_difference,
            abs(self.reference_target_loss - self.transformed_target_loss),
            abs_tol=1e-12,
        ):
            raise ValueError("V3.11 representation pair loss does not recompute")
        return self


class RepresentationTopologyReportV311(StrictModel):
    schema_version: Literal["3.11"] = "3.11"
    report_id: Identifier
    phase: Literal["development", "confirmation"]
    private_spec_hash: Sha256
    public_protocol_hash: Sha256
    public_pack_hash: Sha256
    private_pack_hash: Sha256
    bundle_hash: Sha256
    source_v310_evolution_hash: Sha256
    case_results: list[PrivateTopologyCaseResultV311]
    pair_results: list[RepresentationPairResultV311]
    performance_case_count: Annotated[int, Field(ge=1)]
    quality_case_count: Annotated[int, Field(ge=0)]
    baseline_coverage: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    candidate_coverage: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    baseline_mean_target_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    candidate_mean_target_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    paired_mean_improvement: Annotated[float, Field(allow_inf_nan=False)]
    paired_improvement_ci_lower: Annotated[float, Field(allow_inf_nan=False)]
    paired_improvement_ci_upper: Annotated[float, Field(allow_inf_nan=False)]
    topology_accuracy: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    open_set_case_count: Annotated[int, Field(ge=1)]
    open_set_abstention_rate: Annotated[
        float, Field(ge=0, le=1, allow_inf_nan=False)
    ]
    pair_topology_consistency: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    maximum_pair_loss_difference: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    mean_loss_by_mechanism_baseline: dict[MechanismV311, Annotated[float, Field(ge=0, allow_inf_nan=False)]]
    mean_loss_by_mechanism_candidate: dict[MechanismV311, Annotated[float, Field(ge=0, allow_inf_nan=False)]]
    mean_loss_by_representation_candidate: dict[RepresentationV311, Annotated[float, Field(ge=0, allow_inf_nan=False)]]
    candidate_selection_counts: dict[str, Annotated[int, Field(ge=0)]]
    material_negative_transfer_count: Annotated[int, Field(ge=0)]
    material_negative_transfer_upper_95: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    public_execution_private_blind: Literal[True] = True
    gates: dict[Identifier, bool]
    ready_for_next_confirmation: bool
    task_router_permitted: Literal[False] = False
    model_qualification_permitted: Literal[False] = False
    real_world_execution_permitted: Literal[False] = False
    status: Literal[
        "representation_topology_development_diagnostic_v311",
        "representation_topology_confirmed_v311",
        "representation_topology_refuted_v311",
    ]
    created_at: datetime
    report_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_report(self) -> "RepresentationTopologyReportV311":
        _assert_timezone(self.created_at, "created_at")
        expected_ready = self.phase == "confirmation" and all(self.gates.values())
        if self.phase == "development":
            expected_status = "representation_topology_development_diagnostic_v311"
        elif expected_ready:
            expected_status = "representation_topology_confirmed_v311"
        else:
            expected_status = "representation_topology_refuted_v311"
        if self.ready_for_next_confirmation != expected_ready or self.status != expected_status:
            raise ValueError("V3.11 report status disagrees with phase/gates")
        if set(self.mean_loss_by_mechanism_candidate) != set(MECHANISMS_V311):
            raise ValueError("V3.11 candidate mechanism report incomplete")
        if set(self.mean_loss_by_representation_candidate) != set(REPRESENTATIONS_V311):
            raise ValueError("V3.11 representation report incomplete")
        if self.report_hash and self.report_hash != self.content_hash():
            raise ValueError("report_hash does not match V3.11 report")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "report_hash")

    def assert_sealed(self) -> None:
        if not self.report_hash or self.report_hash != self.content_hash():
            raise ValueError("V3.11 report is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "RepresentationTopologyReportV311":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"report_hash"}),
            report_hash=draft.content_hash(),
        )


def _selected_challenge_v311(
    receipt: TopologyCaseReceiptV311,
    decision: TopologyDecisionV311,
) -> TopologyChallengeReceiptV311 | None:
    if decision.selected_challenge_hash is None:
        return None
    matches = [
        item for item in receipt.challenges
        if item.challenge_hash == decision.selected_challenge_hash
    ]
    if len(matches) != 1:
        raise ValueError("V3.11 decision did not bind one challenge")
    return matches[0]


def _private_loss_v311(
    private_case: PrivateTopologyCaseV311,
    challenge: TopologyChallengeReceiptV311 | None,
    protocol: PublicTopologyProtocolV311,
) -> float:
    if challenge is None:
        return protocol.unresolved_loss
    times = private_case.public_case.trajectories[0].times
    losses = []
    for initial, truth in zip(
        private_case.private_probe_initials,
        private_case.private_probe_truths,
        strict=True,
    ):
        try:
            prediction = _simulate_model_v311(challenge.final_model, initial, times)
            losses.append(trajectory_nrmse(truth, prediction))
        except RuntimeError:
            losses.append(protocol.unresolved_loss)
    return float(np.mean(losses))


def _bootstrap_ci_v311(
    values: np.ndarray,
    spec: PrivateTopologyWorldPackSpecV311,
) -> tuple[float, float]:
    random = np.random.default_rng(spec.bootstrap_seed)
    indices = random.integers(
        0, len(values), size=(spec.bootstrap_replicates, len(values))
    )
    means = np.mean(values[indices], axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def evaluate_topology_discovery_v311(
    private_spec: PrivateTopologyWorldPackSpecV311,
    public_protocol: PublicTopologyProtocolV311,
    private_pack: PrivateTopologyWorldPackV311,
    bundle: TopologyDiscoveryBundleV311,
    *,
    evaluated_at: datetime,
) -> RepresentationTopologyReportV311:
    for artifact in (private_spec, public_protocol, private_pack, bundle):
        artifact.assert_sealed()
    if (
        private_pack.spec_hash != private_spec.spec_hash
        or private_pack.public_pack_hash != bundle.public_pack_hash
        or bundle.public_protocol_hash != public_protocol.protocol_hash
    ):
        raise ValueError("V3.11 evaluator artifact binding differs")
    receipt_by_id = {item.case_id: item for item in bundle.case_receipts}
    results: list[PrivateTopologyCaseResultV311] = []
    quality_count = 0
    baseline_selected = 0
    candidate_selected = 0
    matrix_complete = True
    private_blind = True
    baseline_by_mechanism: dict[str, list[float]] = defaultdict(list)
    candidate_by_mechanism: dict[str, list[float]] = defaultdict(list)
    candidate_by_representation: dict[str, list[float]] = defaultdict(list)
    selection_counts: dict[str, int] = defaultdict(int)
    for private_case in private_pack.cases:
        case_id = private_case.public_case.case_id
        receipt = receipt_by_id[case_id]
        if receipt.quality_flags != private_case.public_case.quality_flags:
            private_blind = False
        if receipt.quality_flags:
            quality_count += 1
            if receipt.challenges:
                matrix_complete = False
            continue
        expected_count = {1: 2, 2: 5, 3: 8}[
            len(private_case.public_case.state_names)
        ]
        if len(receipt.challenges) != expected_count:
            matrix_complete = False
        private_blind = private_blind and all(
            not item.private_values_used and not item.semantic_state_labels_used
            for item in receipt.challenges
        ) and not receipt.candidate_decision.private_values_used
        baseline_challenge = _selected_challenge_v311(
            receipt, receipt.baseline_decision
        )
        candidate_challenge = _selected_challenge_v311(
            receipt, receipt.candidate_decision
        )
        baseline_selected += int(baseline_challenge is not None)
        candidate_selected += int(candidate_challenge is not None)
        baseline_loss = _private_loss_v311(
            private_case, baseline_challenge, public_protocol
        )
        candidate_loss = _private_loss_v311(
            private_case, candidate_challenge, public_protocol
        )
        candidate_topology = receipt.candidate_decision.topology_hypothesis
        if candidate_topology is not None:
            selection_counts[candidate_topology] += 1
        baseline_by_mechanism[private_case.mechanism].append(baseline_loss)
        candidate_by_mechanism[private_case.mechanism].append(candidate_loss)
        candidate_by_representation[private_case.representation].append(candidate_loss)
        results.append(PrivateTopologyCaseResultV311(
            case_id=case_id,
            hidden_pair_id=private_case.hidden_pair_id,
            mechanism=private_case.mechanism,
            representation=private_case.representation,
            baseline_topology=(
                baseline_challenge.topology if baseline_challenge else None
            ),
            candidate_topology=candidate_topology,
            baseline_target_loss=baseline_loss,
            candidate_target_loss=candidate_loss,
            candidate_improvement=baseline_loss - candidate_loss,
            topology_correct=(
                candidate_topology == EXPECTED_TOPOLOGY_V311[private_case.mechanism]
            ),
            material_negative_transfer=(
                candidate_loss - baseline_loss
                > private_spec.material_negative_transfer
            ),
        ))
    by_pair: dict[str, list[PrivateTopologyCaseResultV311]] = defaultdict(list)
    for result in results:
        by_pair[result.hidden_pair_id].append(result)
    pair_results: list[RepresentationPairResultV311] = []
    for pair_id, pair in by_pair.items():
        if len(pair) != 2:
            raise ValueError("V3.11 representation pair is incomplete")
        reference = next(
            item for item in pair if item.representation == "anonymous_reference"
        )
        transformed = next(
            item for item in pair
            if item.representation == "anonymous_scaled_permuted"
        )
        pair_results.append(RepresentationPairResultV311(
            hidden_pair_id=pair_id,
            mechanism=reference.mechanism,
            reference_case_id=reference.case_id,
            transformed_case_id=transformed.case_id,
            reference_topology=reference.candidate_topology,
            transformed_topology=transformed.candidate_topology,
            topology_consistent=(
                reference.candidate_topology == transformed.candidate_topology
            ),
            reference_target_loss=reference.candidate_target_loss,
            transformed_target_loss=transformed.candidate_target_loss,
            absolute_target_loss_difference=abs(
                reference.candidate_target_loss - transformed.candidate_target_loss
            ),
        ))
    baseline_losses = np.asarray([item.baseline_target_loss for item in results])
    candidate_losses = np.asarray([item.candidate_target_loss for item in results])
    improvements = baseline_losses - candidate_losses
    ci_lower, ci_upper = _bootstrap_ci_v311(improvements, private_spec)
    negatives = sum(item.material_negative_transfer for item in results)
    negative_upper = (
        float(beta.ppf(0.95, negatives + 1, len(results) - negatives))
        if len(results) > negatives else 1.0
    )
    baseline_mechanism = {
        mechanism: float(np.mean(baseline_by_mechanism[mechanism]))
        for mechanism in MECHANISMS_V311
    }
    candidate_mechanism = {
        mechanism: float(np.mean(candidate_by_mechanism[mechanism]))
        for mechanism in MECHANISMS_V311
    }
    candidate_representation = {
        representation: float(np.mean(candidate_by_representation[representation]))
        for representation in REPRESENTATIONS_V311
    }
    topology_accuracy = float(np.mean([item.topology_correct for item in results]))
    open_set_results = [
        item for item in results if item.mechanism == "pendulum_open_set"
    ]
    open_set_abstention_rate = float(np.mean([
        item.candidate_topology is None for item in open_set_results
    ]))
    pair_consistency = float(np.mean([
        item.topology_consistent for item in pair_results
    ]))
    max_pair_difference = max(
        item.absolute_target_loss_difference for item in pair_results
    )
    mechanism_nonregression = all(
        candidate_mechanism[mechanism]
        <= baseline_mechanism[mechanism]
        + private_spec.maximum_mechanism_regression
        for mechanism in MECHANISMS_V311
    )
    gates = {
        "public_quality_partition_complete": (
            quality_count == private_spec.expected_quality_case_count
        ),
        "candidate_matrix_complete": matrix_complete,
        "public_execution_private_blind": private_blind,
        "supported_candidate_coverage": (
            sum(
                item.candidate_topology is not None
                for item in results
                if item.mechanism != "pendulum_open_set"
            ) / sum(
                item.mechanism != "pendulum_open_set" for item in results
            )
            >= private_spec.minimum_candidate_coverage
        ),
        "topology_accuracy": (
            topology_accuracy >= private_spec.minimum_topology_accuracy
        ),
        "open_set_abstention": (
            open_set_abstention_rate
            >= private_spec.minimum_open_set_abstention_rate
        ),
        "paired_topology_consistency": (
            pair_consistency >= private_spec.minimum_pair_topology_consistency
        ),
        "paired_prediction_invariance": (
            max_pair_difference <= private_spec.maximum_pair_loss_difference
        ),
        "paired_improvement_ci_lower_positive": ci_lower > 0.0,
        "mechanism_non_regression": mechanism_nonregression,
        "transformed_representation_non_regression": (
            candidate_representation["anonymous_scaled_permuted"]
            <= candidate_representation["anonymous_reference"]
            + private_spec.maximum_transformed_representation_regression
        ),
        "material_negative_transfer_controlled": (
            negative_upper <= private_spec.maximum_negative_transfer_upper_95
        ),
        "no_task_router_or_real_world_execution": True,
    }
    all_selection_counts = {
        topology: selection_counts.get(topology, 0)
        for topology in TOPOLOGIES_V311
    }
    ready = private_spec.phase == "confirmation" and all(gates.values())
    status: Literal[
        "representation_topology_development_diagnostic_v311",
        "representation_topology_confirmed_v311",
        "representation_topology_refuted_v311",
    ]
    if private_spec.phase == "development":
        status = "representation_topology_development_diagnostic_v311"
    elif ready:
        status = "representation_topology_confirmed_v311"
    else:
        status = "representation_topology_refuted_v311"
    return RepresentationTopologyReportV311.seal(
        report_id=f"report_{private_spec.experiment_id}",
        phase=private_spec.phase,
        private_spec_hash=private_spec.spec_hash,
        public_protocol_hash=public_protocol.protocol_hash,
        public_pack_hash=private_pack.public_pack_hash,
        private_pack_hash=private_pack.pack_hash,
        bundle_hash=bundle.bundle_hash,
        source_v310_evolution_hash=private_spec.source_v310_evolution_hash,
        case_results=results,
        pair_results=pair_results,
        performance_case_count=len(results),
        quality_case_count=quality_count,
        baseline_coverage=baseline_selected / len(results),
        candidate_coverage=(
            sum(
                item.candidate_topology is not None
                for item in results
                if item.mechanism != "pendulum_open_set"
            ) / sum(item.mechanism != "pendulum_open_set" for item in results)
        ),
        baseline_mean_target_loss=float(np.mean(baseline_losses)),
        candidate_mean_target_loss=float(np.mean(candidate_losses)),
        paired_mean_improvement=float(np.mean(improvements)),
        paired_improvement_ci_lower=ci_lower,
        paired_improvement_ci_upper=ci_upper,
        topology_accuracy=topology_accuracy,
        open_set_case_count=len(open_set_results),
        open_set_abstention_rate=open_set_abstention_rate,
        pair_topology_consistency=pair_consistency,
        maximum_pair_loss_difference=max_pair_difference,
        mean_loss_by_mechanism_baseline=baseline_mechanism,
        mean_loss_by_mechanism_candidate=candidate_mechanism,
        mean_loss_by_representation_candidate=candidate_representation,
        candidate_selection_counts=all_selection_counts,
        material_negative_transfer_count=negatives,
        material_negative_transfer_upper_95=negative_upper,
        gates=gates,
        ready_for_next_confirmation=ready,
        status=status,
        created_at=evaluated_at,
    )


class RepresentationTopologyManifestV311(StrictModel):
    schema_version: Literal["3.11"] = "3.11"
    run_id: Identifier
    phase: Literal["development", "confirmation"]
    artifact_refs: list[ArtifactRef] = Field(min_length=8, max_length=8)
    terminal_status: Literal[
        "representation_topology_development_diagnostic_v311",
        "representation_topology_confirmed_v311",
        "representation_topology_refuted_v311",
    ]
    created_at: datetime
    manifest_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_manifest(self) -> "RepresentationTopologyManifestV311":
        _assert_timezone(self.created_at, "created_at")
        if len({item.kind for item in self.artifact_refs}) != 8:
            raise ValueError("V3.11 manifest artifact kinds differ")
        if self.manifest_hash and self.manifest_hash != self.content_hash():
            raise ValueError("manifest_hash does not match V3.11 manifest")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "manifest_hash")

    def assert_sealed(self) -> None:
        if not self.manifest_hash or self.manifest_hash != self.content_hash():
            raise ValueError("V3.11 manifest is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "RepresentationTopologyManifestV311":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"manifest_hash"}),
            manifest_hash=draft.content_hash(),
        )


@dataclass(frozen=True)
class RepresentationTopologyOutcomeV311:
    store: RunStore
    public_pack: PublicTopologyWorldPackV311
    private_pack: PrivateTopologyWorldPackV311
    bundle: TopologyDiscoveryBundleV311
    report: RepresentationTopologyReportV311
    manifest: RepresentationTopologyManifestV311


def _load_development_report_v311(
    run_directory: str | Path,
    *,
    source_v310_run_directory: str | Path,
) -> RepresentationTopologyReportV311:
    if not verify_representation_topology_run_v311(
        run_directory,
        source_v310_run_directory=source_v310_run_directory,
        development_run_directory=None,
    ):
        raise ValueError("V3.11 development run did not independently verify")
    store = RunStore.open_existing(run_directory)
    refs = _committed_refs_v311(store)
    report = _load_one_v311(
        store,
        refs,
        "representation_topology_report_v311",
        RepresentationTopologyReportV311,
    )
    if report.phase != "development":
        raise ValueError("V3.11 development lineage has wrong phase")
    return report


def run_representation_topology_worldpack_v311(
    output_root: str | Path,
    *,
    source_v310_run_directory: str | Path,
    development_run_directory: str | Path | None = None,
    evidence: RepresentationMethodEvidenceV311,
    policy: TopologyDiscoveryPolicyV311,
    public_protocol: PublicTopologyProtocolV311,
    private_spec: PrivateTopologyWorldPackSpecV311,
    evaluated_at: datetime | None = None,
    run_id: str | None = None,
) -> RepresentationTopologyOutcomeV311:
    source = _load_source_v310(source_v310_run_directory)
    development_report = None
    if private_spec.phase == "confirmation":
        if development_run_directory is None:
            raise ValueError("V3.11 confirmation requires development run")
        development_report = _load_development_report_v311(
            development_run_directory,
            source_v310_run_directory=source_v310_run_directory,
        )
    for artifact in (evidence, policy, public_protocol, private_spec):
        artifact.assert_sealed()
    if (
        source.evolution_hash != policy.source_v310_evolution_hash
        or source.evolution_hash != private_spec.source_v310_evolution_hash
        or evidence.evidence_hash != policy.method_evidence_hash
        or evidence.evidence_hash != public_protocol.method_evidence_hash
        or policy.policy_hash != public_protocol.policy_hash
        or public_protocol.protocol_hash != private_spec.public_protocol_hash
        or private_spec.frozen_at < public_protocol.frozen_at
        or (
            private_spec.phase == "confirmation"
            and (
                development_report.report_hash
                != private_spec.development_report_hash
                or private_spec.frozen_at < development_report.created_at
            )
        )
    ):
        raise ValueError("V3.11 frozen lineage binding differs")
    at = evaluated_at or datetime.now(timezone.utc)
    if at < private_spec.frozen_at:
        raise ValueError("V3.11 evaluation predates frozen private spec")
    wall_now = datetime.now(timezone.utc)
    if (
        private_spec.frozen_at > wall_now + timedelta(minutes=5)
        or at > wall_now + timedelta(minutes=5)
    ):
        raise ValueError("V3.11 audit timestamp is implausibly in the future")
    store = RunStore(
        output_root,
        run_id=run_id or f"representation-topology-v311-{uuid4().hex[:10]}",
    )
    refs = [
        store.put_artifact("representation_method_evidence_v311", evidence),
        store.put_artifact("topology_discovery_policy_v311", policy),
        store.put_artifact("public_topology_protocol_v311", public_protocol),
        store.put_artifact("private_topology_worldpack_spec_v311", private_spec),
    ]
    store.emit("representation_topology_v311_protocol_frozen_before_private_pack", {
        "phase": private_spec.phase,
        "private_spec_hash": private_spec.spec_hash,
        "public_protocol_hash": public_protocol.protocol_hash,
        "source_v310_evolution_hash": source.evolution_hash,
        "private_pack_not_passed_to_generator": True,
        "development_report_hash": private_spec.development_report_hash,
    })
    public_pack, private_pack = generate_topology_worldpacks_v311(
        private_spec, public_protocol, generated_at=at
    )
    bundle = execute_topology_discovery_v311(
        public_protocol, public_pack, policy, executed_at=at
    )
    report = evaluate_topology_discovery_v311(
        private_spec,
        public_protocol,
        private_pack,
        bundle,
        evaluated_at=at,
    )
    refs.extend([
        store.put_artifact("public_topology_worldpack_v311", public_pack),
        store.put_artifact("private_topology_worldpack_v311", private_pack),
        store.put_artifact("topology_discovery_bundle_v311", bundle),
        store.put_artifact("representation_topology_report_v311", report),
    ])
    manifest = RepresentationTopologyManifestV311.seal(
        run_id=store.run_id,
        phase=private_spec.phase,
        artifact_refs=refs,
        terminal_status=report.status,
        created_at=at,
    )
    manifest_ref = store.put_artifact(
        "representation_topology_manifest_v311", manifest
    )
    store.emit("representation_topology_v311_worldpack_adjudicated", {
        "manifest_ref": manifest_ref.model_dump(mode="json")
    })
    if not verify_representation_topology_run_v311(
        store.run_directory,
        source_v310_run_directory=source_v310_run_directory,
        development_run_directory=development_run_directory,
    ):
        raise RuntimeError("V3.11 run failed independent verification")
    return RepresentationTopologyOutcomeV311(
        store, public_pack, private_pack, bundle, report, manifest
    )


def verify_representation_topology_run_v311(
    run_directory: str | Path,
    *,
    source_v310_run_directory: str | Path,
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
        refs = _committed_refs_v311(store)
        if len(refs) != 9:
            return False
        for ref in refs:
            store.load_artifact(ref)
        evidence = _load_one_v311(
            store, refs, "representation_method_evidence_v311",
            RepresentationMethodEvidenceV311,
        )
        policy = _load_one_v311(
            store, refs, "topology_discovery_policy_v311",
            TopologyDiscoveryPolicyV311,
        )
        public_protocol = _load_one_v311(
            store, refs, "public_topology_protocol_v311",
            PublicTopologyProtocolV311,
        )
        private_spec = _load_one_v311(
            store, refs, "private_topology_worldpack_spec_v311",
            PrivateTopologyWorldPackSpecV311,
        )
        public_pack = _load_one_v311(
            store, refs, "public_topology_worldpack_v311",
            PublicTopologyWorldPackV311,
        )
        private_pack = _load_one_v311(
            store, refs, "private_topology_worldpack_v311",
            PrivateTopologyWorldPackV311,
        )
        bundle = _load_one_v311(
            store, refs, "topology_discovery_bundle_v311",
            TopologyDiscoveryBundleV311,
        )
        report = _load_one_v311(
            store, refs, "representation_topology_report_v311",
            RepresentationTopologyReportV311,
        )
        manifest = _load_one_v311(
            store, refs, "representation_topology_manifest_v311",
            RepresentationTopologyManifestV311,
        )
        for artifact in (
            evidence, policy, public_protocol, private_spec,
            public_pack, private_pack, bundle, report, manifest,
        ):
            artifact.assert_sealed()
        wall_now = datetime.now(timezone.utc)
        if (
            private_spec.frozen_at < public_protocol.frozen_at
            or report.created_at < private_spec.frozen_at
            or private_spec.frozen_at > wall_now + timedelta(minutes=5)
            or report.created_at > wall_now + timedelta(minutes=5)
        ):
            return False
        development_report = None
        if private_spec.phase == "confirmation":
            if development_run_directory is None:
                return False
            if Path(development_run_directory).resolve() == Path(run_directory).resolve():
                return False
            development_report = _load_development_report_v311(
                development_run_directory,
                source_v310_run_directory=source_v310_run_directory,
            )
        source = _load_source_v310(source_v310_run_directory)
        if (
            source.evolution_hash != policy.source_v310_evolution_hash
            or source.evolution_hash != private_spec.source_v310_evolution_hash
            or evidence.evidence_hash != policy.method_evidence_hash
            or evidence.evidence_hash != public_protocol.method_evidence_hash
            or policy.policy_hash != public_protocol.policy_hash
            or public_protocol.protocol_hash != private_spec.public_protocol_hash
            or private_spec.frozen_at < public_protocol.frozen_at
            or report.created_at < private_spec.frozen_at
            or (
                private_spec.phase == "confirmation"
                and (
                    development_report.report_hash
                    != private_spec.development_report_hash
                    or private_spec.frozen_at < development_report.created_at
                )
            )
        ):
            return False
        regenerated_public, regenerated_private = generate_topology_worldpacks_v311(
            private_spec, public_protocol, generated_at=private_pack.generated_at
        )
        if (
            regenerated_public.public_pack_hash != public_pack.public_pack_hash
            or regenerated_private.pack_hash != private_pack.pack_hash
        ):
            return False
        recomputed_bundle = execute_topology_discovery_v311(
            public_protocol, public_pack, policy, executed_at=bundle.created_at
        )
        if recomputed_bundle.bundle_hash != bundle.bundle_hash:
            return False
        recomputed_report = evaluate_topology_discovery_v311(
            private_spec,
            public_protocol,
            private_pack,
            bundle,
            evaluated_at=report.created_at,
        )
        if recomputed_report.report_hash != report.report_hash:
            return False
        if (
            manifest.phase != private_spec.phase
            or manifest.terminal_status != report.status
            or [item.model_dump(mode="json") for item in manifest.artifact_refs]
            != [item.model_dump(mode="json") for item in refs[:8]]
        ):
            return False
        event_types = [event["event_type"] for event in events]
        freeze_index = event_types.index(
            "representation_topology_v311_protocol_frozen_before_private_pack"
        )
        private_index = next(
            index for index, event in enumerate(events)
            if event["event_type"] == "artifact_committed"
            and event["payload"]["kind"] == "private_topology_worldpack_v311"
        )
        return freeze_index < private_index
    except (
        OSError, RuntimeError, ValueError, KeyError, TypeError,
        json.JSONDecodeError, FloatingPointError, AttributeError,
        np.linalg.LinAlgError,
    ):
        return False
