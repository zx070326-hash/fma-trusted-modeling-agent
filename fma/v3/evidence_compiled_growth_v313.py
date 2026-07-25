from __future__ import annotations

import hashlib
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
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from scipy.stats import beta

from fma.hashing import sha256_value
from fma.schemas import ArtifactRef, StrictModel
from fma.storage import RunStore
from fma.v2.dynamics_ir import trajectory_nrmse
from fma.v2.schemas import Identifier, Sha256, _assert_timezone

from .model_challenge_v37 import _hash_without
from .evidence_concept_compiler_v313 import (
    CompiledConceptV313,
    ConceptEvidenceBundleV313,
    ConceptExperienceStoreV313,
    ConceptPackageV313,
    append_concept_experience_event_v313,
    compile_concept_package_v313,
    default_concept_evidence_v313,
    default_concept_packages_v313,
    empty_concept_experience_store_v313,
    evaluate_operator_ast_v313,
)
from .open_set_concept_evolution_v312 import (
    ConceptEvolutionManifestV312,
    ConceptEvolutionReportV312,
    verify_concept_evolution_run_v312,
)
from . import open_set_concept_evolution_v312 as v312_module


DEVELOPMENT_SEEDS_V313 = (31019, 31081, 31151, 31219, 31283, 31349)
CONFIRMATION_SEEDS_V313 = (
    32017, 32077, 32147, 32221, 32287, 32363, 32429, 32501,
)
MechanismV313 = Literal[
    "gompertz_growth",
    "richards_growth",
    "monod_net_growth",
]
RepresentationV313 = Literal["anonymous_reference", "anonymous_scaled"]
MECHANISMS_V313: tuple[MechanismV313, ...] = (
    "gompertz_growth",
    "richards_growth",
    "monod_net_growth",
)
REPRESENTATIONS_V313: tuple[RepresentationV313, ...] = (
    "anonymous_reference",
    "anonymous_scaled",
)
EXPECTED_CONCEPT_V313: dict[MechanismV313, str] = {
    "gompertz_growth": "log_capacity_growth",
    "richards_growth": "generalized_capacity_growth",
    "monod_net_growth": "hyperbolic_net_growth",
}
BASELINE_DEGREES_V313 = (1, 2, 3, 4)


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
        raise ValueError(f"V3.13 expected one {kind}")
    return model.model_validate(store.load_artifact(matches[0]))


class VerifiedV312LineageReceiptV313(StrictModel):
    schema_version: Literal["3.13"] = "3.13"
    receipt_id: Identifier
    source_run_id: Identifier
    source_report_hash: Sha256
    source_manifest_hash: Sha256
    source_event_head_hash: Sha256
    source_artifact_refs: list[ArtifactRef] = Field(min_length=10, max_length=10)
    source_verifier_file_hash: Sha256
    full_replay_verified: Literal[True] = True
    verified_at: datetime
    receipt_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_receipt(self) -> "VerifiedV312LineageReceiptV313":
        _assert_timezone(self.verified_at, "verified_at")
        if self.receipt_hash and self.receipt_hash != self.content_hash():
            raise ValueError("V3.13 lineage receipt hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "receipt_hash")

    def assert_sealed(self) -> None:
        if not self.receipt_hash or self.receipt_hash != self.content_hash():
            raise ValueError("V3.13 lineage receipt is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "VerifiedV312LineageReceiptV313":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"receipt_hash"}),
            receipt_hash=draft.content_hash(),
        )


def build_verified_v312_lineage_receipt_v313(
    source_run_directory: str | Path,
    *,
    source_v311_run_directory: str | Path,
    source_development_run_directory: str | Path,
    verified_at: datetime | None = None,
) -> VerifiedV312LineageReceiptV313:
    if not verify_concept_evolution_run_v312(
        source_run_directory,
        source_run_directory=source_v311_run_directory,
        development_run_directory=source_development_run_directory,
    ):
        raise ValueError("V3.13 source V3.12 did not fully replay")
    store = RunStore.open_existing(source_run_directory)
    refs = _committed_refs(store)
    report = _load_one(
        store, refs, "concept_evolution_report_v312", ConceptEvolutionReportV312
    )
    manifest = _load_one(
        store, refs, "concept_evolution_manifest_v312", ConceptEvolutionManifestV312
    )
    if report.status != "open_set_concepts_admitted_v312":
        raise ValueError("V3.13 source V3.12 is not admitted")
    last_event = json.loads(
        store.event_path.read_text(encoding="utf-8").splitlines()[-1]
    )
    return VerifiedV312LineageReceiptV313.seal(
        receipt_id="verified_v312_lineage_for_v313",
        source_run_id=store.run_id,
        source_report_hash=report.report_hash,
        source_manifest_hash=manifest.manifest_hash,
        source_event_head_hash=last_event["event_hash"],
        source_artifact_refs=refs,
        source_verifier_file_hash=_file_sha256(v312_module.__file__),
        verified_at=verified_at or datetime.now(timezone.utc),
    )


def verify_v312_lineage_receipt_v313(
    receipt: VerifiedV312LineageReceiptV313,
    source_run_directory: str | Path,
) -> bool:
    try:
        receipt.assert_sealed()
        if receipt.verified_at > datetime.now(timezone.utc) + timedelta(minutes=5):
            return False
        if _file_sha256(v312_module.__file__) != receipt.source_verifier_file_hash:
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
            store, refs, "concept_evolution_report_v312", ConceptEvolutionReportV312
        )
        manifest = _load_one(
            store, refs, "concept_evolution_manifest_v312", ConceptEvolutionManifestV312
        )
        return (
            report.status == "open_set_concepts_admitted_v312"
            and report.report_hash == receipt.source_report_hash
            and manifest.manifest_hash == receipt.source_manifest_hash
        )
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


class CompiledConceptEntryV313(StrictModel):
    package: ConceptPackageV313
    compiled: CompiledConceptV313

    @model_validator(mode="after")
    def validate_entry(self) -> "CompiledConceptEntryV313":
        self.package.assert_sealed()
        self.compiled.assert_sealed()
        if self.package.package_hash != self.compiled.package_hash:
            raise ValueError("V3.13 compiled library entry differs")
        return self


class CompiledConceptLibraryV313(StrictModel):
    schema_version: Literal["3.13"] = "3.13"
    library_id: Identifier
    evidence_hash: Sha256
    entries: list[CompiledConceptEntryV313] = Field(min_length=4, max_length=4)
    arbitrary_code_available: Literal[False] = False
    library_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_library(self) -> "CompiledConceptLibraryV313":
        ids = [item.package.concept_id for item in self.entries]
        if ids != [
            "log_capacity_growth",
            "generalized_capacity_growth",
            "hyperbolic_net_growth",
            "affine_rate_decoy",
        ]:
            raise ValueError("V3.13 compiled concept order differs")
        if any(item.package.evidence_hash != self.evidence_hash for item in self.entries):
            raise ValueError("V3.13 library evidence binding differs")
        if self.library_hash and self.library_hash != self.content_hash():
            raise ValueError("V3.13 compiled library hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "library_hash")

    def assert_sealed(self) -> None:
        if not self.library_hash or self.library_hash != self.content_hash():
            raise ValueError("V3.13 compiled library is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "CompiledConceptLibraryV313":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"library_hash"}),
            library_hash=draft.content_hash(),
        )


def compile_default_concept_library_v313(
    evidence: ConceptEvidenceBundleV313,
) -> CompiledConceptLibraryV313:
    packages = default_concept_packages_v313(evidence)
    return CompiledConceptLibraryV313.seal(
        library_id="source_grounded_growth_concepts_v313",
        evidence_hash=evidence.evidence_hash,
        entries=[
            CompiledConceptEntryV313(
                package=package,
                compiled=compile_concept_package_v313(package, evidence),
            )
            for package in packages
        ],
    )


class EvidenceConceptPolicyV313(StrictModel):
    schema_version: Literal["3.13"] = "3.13"
    policy_id: Identifier
    evidence_hash: Sha256
    lineage_receipt_hash: Sha256
    library_hash: Sha256
    baseline_degrees: list[int] = Field(min_length=4, max_length=4)
    candidate_package_hashes: list[Sha256] = Field(min_length=4, max_length=4)
    expression_evaluations_per_arm: Literal[4] = 4
    same_candidate_library_for_every_case: Literal[True] = True
    arbitrary_code_execution_permitted: Literal[False] = False
    source_custom_operator_permitted: Literal[False] = False
    private_mechanism_visible: Literal[False] = False
    private_representation_visible: Literal[False] = False
    private_probe_visible: Literal[False] = False
    private_loss_visible: Literal[False] = False
    public_score_can_admit_concept: Literal[False] = False
    task_router_permitted: Literal[False] = False
    real_world_execution_permitted: Literal[False] = False
    policy_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_policy(self) -> "EvidenceConceptPolicyV313":
        if self.baseline_degrees != list(BASELINE_DEGREES_V313):
            raise ValueError("V3.13 baseline degrees differ")
        if self.policy_hash and self.policy_hash != self.content_hash():
            raise ValueError("V3.13 policy hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "policy_hash")

    def assert_sealed(self) -> None:
        if not self.policy_hash or self.policy_hash != self.content_hash():
            raise ValueError("V3.13 policy is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "EvidenceConceptPolicyV313":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"policy_hash"}),
            policy_hash=draft.content_hash(),
        )


def default_evidence_concept_policy_v313(
    evidence: ConceptEvidenceBundleV313,
    lineage: VerifiedV312LineageReceiptV313,
    library: CompiledConceptLibraryV313,
) -> EvidenceConceptPolicyV313:
    for item in (evidence, lineage, library):
        item.assert_sealed()
    return EvidenceConceptPolicyV313.seal(
        policy_id="evidence_compiled_same_dimension_growth_v313",
        evidence_hash=evidence.evidence_hash,
        lineage_receipt_hash=lineage.receipt_hash,
        library_hash=library.library_hash,
        baseline_degrees=list(BASELINE_DEGREES_V313),
        candidate_package_hashes=[
            item.package.package_hash for item in library.entries
        ],
    )


class PublicGrowthProtocolV313(StrictModel):
    schema_version: Literal["3.13"] = "3.13"
    protocol_id: Identifier
    trajectory_points: Literal[81] = 81
    time_step: Literal[0.05] = 0.05
    public_trajectory_count: Literal[4] = 4
    fit_trajectory_count: Literal[2] = 2
    validation_trajectory_index: Literal[2] = 2
    challenge_trajectory_index: Literal[3] = 3
    savgol_window: Literal[11] = 11
    savgol_polynomial: Literal[3] = 3
    ridge_alpha: Literal[1e-8] = 1e-8
    complexity_penalty: Literal[0.001] = 0.001
    maximum_candidate_public_score: Literal[1.0] = 1.0
    minimum_selected_concept_influence: Literal[0.001] = 0.001
    unresolved_loss: Literal[10.0] = 10.0
    evidence_hash: Sha256
    lineage_receipt_hash: Sha256
    policy_hash: Sha256
    library_hash: Sha256
    frozen_at: datetime
    protocol_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_protocol(self) -> "PublicGrowthProtocolV313":
        _assert_timezone(self.frozen_at, "frozen_at")
        if self.protocol_hash and self.protocol_hash != self.content_hash():
            raise ValueError("V3.13 public protocol hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "protocol_hash")

    def assert_sealed(self) -> None:
        if not self.protocol_hash or self.protocol_hash != self.content_hash():
            raise ValueError("V3.13 public protocol is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "PublicGrowthProtocolV313":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"protocol_hash"}),
            protocol_hash=draft.content_hash(),
        )


def default_public_growth_protocol_v313(
    evidence: ConceptEvidenceBundleV313,
    lineage: VerifiedV312LineageReceiptV313,
    policy: EvidenceConceptPolicyV313,
    library: CompiledConceptLibraryV313,
    *,
    frozen_at: datetime | None = None,
) -> PublicGrowthProtocolV313:
    for item in (evidence, lineage, policy, library):
        item.assert_sealed()
    return PublicGrowthProtocolV313.seal(
        protocol_id="same_dimension_evidence_concept_protocol_v313",
        evidence_hash=evidence.evidence_hash,
        lineage_receipt_hash=lineage.receipt_hash,
        policy_hash=policy.policy_hash,
        library_hash=library.library_hash,
        frozen_at=frozen_at or datetime.now(timezone.utc),
    )


class PrivateGrowthWorldPackSpecV313(StrictModel):
    schema_version: Literal["3.13"] = "3.13"
    experiment_id: Identifier
    phase: Literal["development", "confirmation"]
    mechanisms: list[MechanismV313] = Field(min_length=3, max_length=3)
    representations: list[RepresentationV313] = Field(min_length=2, max_length=2)
    seeds: list[int] = Field(min_length=6, max_length=8)
    observation_noise_fraction: Literal[0.001] = 0.001
    calibration_failure_seed_index: Literal[0] = 0
    expected_quality_case_count: Literal[6] = 6
    public_protocol_hash: Sha256
    lineage_receipt_hash: Sha256
    library_hash: Sha256
    development_report_hash: Sha256 | None = None
    development_experience_store_hash: Sha256 | None = None
    bootstrap_replicates: Literal[2000] = 2000
    bootstrap_seed: int
    material_negative_transfer: Literal[0.02] = 0.02
    maximum_negative_transfer_upper_95: Literal[0.1] = 0.1
    minimum_concept_recovery_accuracy: Literal[0.9] = 0.9
    minimum_pair_concept_consistency: Literal[0.9] = 0.9
    maximum_pair_loss_difference: Literal[0.05] = 0.05
    maximum_mechanism_regression: Literal[0.02] = 0.02
    maximum_scaled_representation_regression: Literal[0.02] = 0.02
    frozen_delta: Literal[
        "source_bound_compiler_same_dimension_growth_only"
    ] = "source_bound_compiler_same_dimension_growth_only"
    frozen_at: datetime
    spec_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_spec(self) -> "PrivateGrowthWorldPackSpecV313":
        _assert_timezone(self.frozen_at, "frozen_at")
        if self.mechanisms != list(MECHANISMS_V313):
            raise ValueError("V3.13 mechanisms differ")
        if self.representations != list(REPRESENTATIONS_V313):
            raise ValueError("V3.13 representations differ")
        expected = (
            list(DEVELOPMENT_SEEDS_V313)
            if self.phase == "development" else list(CONFIRMATION_SEEDS_V313)
        )
        if self.seeds != expected:
            raise ValueError("V3.13 seeds differ")
        bindings = (
            self.development_report_hash is not None,
            self.development_experience_store_hash is not None,
        )
        if self.phase == "development" and any(bindings):
            raise ValueError("V3.13 development cannot bind itself")
        if self.phase == "confirmation" and not all(bindings):
            raise ValueError("V3.13 confirmation needs development lineage")
        if self.spec_hash and self.spec_hash != self.content_hash():
            raise ValueError("V3.13 private spec hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "spec_hash")

    def assert_sealed(self) -> None:
        if not self.spec_hash or self.spec_hash != self.content_hash():
            raise ValueError("V3.13 private spec is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "PrivateGrowthWorldPackSpecV313":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"spec_hash"}),
            spec_hash=draft.content_hash(),
        )


def default_private_growth_spec_v313(
    protocol: PublicGrowthProtocolV313,
    lineage: VerifiedV312LineageReceiptV313,
    library: CompiledConceptLibraryV313,
    *,
    phase: Literal["development", "confirmation"],
    development_report_hash: str | None = None,
    development_experience_store_hash: str | None = None,
    frozen_at: datetime | None = None,
) -> PrivateGrowthWorldPackSpecV313:
    for item in (protocol, lineage, library):
        item.assert_sealed()
    return PrivateGrowthWorldPackSpecV313.seal(
        experiment_id=f"evidence_compiled_growth_{phase}_v313",
        phase=phase,
        mechanisms=list(MECHANISMS_V313),
        representations=list(REPRESENTATIONS_V313),
        seeds=(
            list(DEVELOPMENT_SEEDS_V313)
            if phase == "development" else list(CONFIRMATION_SEEDS_V313)
        ),
        public_protocol_hash=protocol.protocol_hash,
        lineage_receipt_hash=lineage.receipt_hash,
        library_hash=library.library_hash,
        development_report_hash=development_report_hash,
        development_experience_store_hash=development_experience_store_hash,
        bootstrap_seed=(3130722 if phase == "development" else 3131722),
        frozen_at=frozen_at or datetime.now(timezone.utc),
    )


class AnonymousGrowthTrajectoryV313(StrictModel):
    schema_version: Literal["3.13"] = "3.13"
    trajectory_id: Identifier
    case_id: Identifier
    times: list[Annotated[float, Field(allow_inf_nan=False)]] = Field(
        min_length=81, max_length=81
    )
    states: list[list[Annotated[float, Field(allow_inf_nan=False)]]] = Field(
        min_length=81, max_length=81
    )
    trajectory_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_trajectory(self) -> "AnonymousGrowthTrajectoryV313":
        if len(self.times) != len(self.states):
            raise ValueError("V3.13 trajectory arrays differ")
        if any(right <= left for left, right in zip(self.times, self.times[1:])):
            raise ValueError("V3.13 trajectory times must increase")
        if any(len(row) != 1 for row in self.states):
            raise ValueError("V3.13 public trajectory must remain one-dimensional")
        if self.trajectory_hash and self.trajectory_hash != self.content_hash():
            raise ValueError("V3.13 trajectory hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "trajectory_hash")

    def assert_sealed(self) -> None:
        if not self.trajectory_hash or self.trajectory_hash != self.content_hash():
            raise ValueError("V3.13 trajectory is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "AnonymousGrowthTrajectoryV313":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"trajectory_hash"}),
            trajectory_hash=draft.content_hash(),
        )


class PublicGrowthCaseV313(StrictModel):
    schema_version: Literal["3.13"] = "3.13"
    case_id: Identifier
    state_names: list[Literal["z0"]] = Field(min_length=1, max_length=1)
    trajectories: list[AnonymousGrowthTrajectoryV313] = Field(
        min_length=4, max_length=4
    )
    quality_flags: list[Identifier]
    semantic_state_labels_available: Literal[False] = False
    mechanism_label_available: Literal[False] = False
    representation_metadata_available: Literal[False] = False
    candidate_library_routing_metadata_available: Literal[False] = False
    public_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_case(self) -> "PublicGrowthCaseV313":
        if self.state_names != ["z0"]:
            raise ValueError("V3.13 state name must be anonymous")
        for item in self.trajectories:
            item.assert_sealed()
            if item.case_id != self.case_id:
                raise ValueError("V3.13 trajectory case differs")
        if self.public_hash and self.public_hash != self.content_hash():
            raise ValueError("V3.13 public case hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "public_hash")

    def assert_sealed(self) -> None:
        if not self.public_hash or self.public_hash != self.content_hash():
            raise ValueError("V3.13 public case is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "PublicGrowthCaseV313":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"public_hash"}),
            public_hash=draft.content_hash(),
        )


class PublicGrowthWorldPackV313(StrictModel):
    schema_version: Literal["3.13"] = "3.13"
    public_protocol_hash: Sha256
    library_hash: Sha256
    cases: list[PublicGrowthCaseV313] = Field(min_length=36, max_length=48)
    public_pack_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_pack(self) -> "PublicGrowthWorldPackV313":
        ids = [item.case_id for item in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("V3.13 public case ids differ")
        for item in self.cases:
            item.assert_sealed()
        if self.public_pack_hash and self.public_pack_hash != self.content_hash():
            raise ValueError("V3.13 public pack hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "public_pack_hash")

    def assert_sealed(self) -> None:
        if not self.public_pack_hash or self.public_pack_hash != self.content_hash():
            raise ValueError("V3.13 public pack is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "PublicGrowthWorldPackV313":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"public_pack_hash"}),
            public_pack_hash=draft.content_hash(),
        )


class PrivateGrowthCaseV313(StrictModel):
    schema_version: Literal["3.13"] = "3.13"
    public_case: PublicGrowthCaseV313
    mechanism: MechanismV313
    representation: RepresentationV313
    expected_concept: Identifier
    hidden_pair_id: Identifier
    hidden_seed: int
    physical_parameters: dict[Identifier, Annotated[float, Field(allow_inf_nan=False)]]
    observed_scale: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    private_probe_initials: list[list[Annotated[float, Field(allow_inf_nan=False)]]] = Field(
        min_length=3, max_length=3
    )
    private_probe_truths: list[
        list[list[Annotated[float, Field(allow_inf_nan=False)]]]
    ] = Field(min_length=3, max_length=3)
    private_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_private(self) -> "PrivateGrowthCaseV313":
        self.public_case.assert_sealed()
        if self.expected_concept != EXPECTED_CONCEPT_V313[self.mechanism]:
            raise ValueError("V3.13 expected concept differs")
        if any(len(item) != 1 for item in self.private_probe_initials):
            raise ValueError("V3.13 private probes must remain one-dimensional")
        if self.private_hash and self.private_hash != self.content_hash():
            raise ValueError("V3.13 private case hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "private_hash")

    def assert_sealed(self) -> None:
        if not self.private_hash or self.private_hash != self.content_hash():
            raise ValueError("V3.13 private case is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "PrivateGrowthCaseV313":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"private_hash"}),
            private_hash=draft.content_hash(),
        )


class PrivateGrowthWorldPackV313(StrictModel):
    schema_version: Literal["3.13"] = "3.13"
    spec_hash: Sha256
    public_pack_hash: Sha256
    cases: list[PrivateGrowthCaseV313] = Field(min_length=36, max_length=48)
    generated_at: datetime
    pack_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_pack(self) -> "PrivateGrowthWorldPackV313":
        _assert_timezone(self.generated_at, "generated_at")
        for item in self.cases:
            item.assert_sealed()
        if self.pack_hash and self.pack_hash != self.content_hash():
            raise ValueError("V3.13 private pack hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "pack_hash")

    def assert_sealed(self) -> None:
        if not self.pack_hash or self.pack_hash != self.content_hash():
            raise ValueError("V3.13 private pack is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "PrivateGrowthWorldPackV313":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"pack_hash"}),
            pack_hash=draft.content_hash(),
        )


def _mechanism_parameters_v313(
    mechanism: MechanismV313,
    seed: int,
) -> tuple[dict[str, float], float]:
    random = Random(seed + 103 * MECHANISMS_V313.index(mechanism))
    if mechanism == "gompertz_growth":
        parameters = {
            "r": random.uniform(0.32, 0.62),
            "K": random.uniform(2.8, 5.2),
        }
        equilibrium = parameters["K"]
    elif mechanism == "richards_growth":
        parameters = {
            "r": random.uniform(0.30, 0.58),
            "K": random.uniform(2.8, 5.2),
            "nu": random.uniform(1.7, 3.1),
        }
        equilibrium = parameters["K"]
    else:
        c = random.uniform(0.16, 0.30)
        b = random.uniform(0.55, 1.25)
        equilibrium = random.uniform(3.2, 5.4)
        parameters = {
            "a": c * (equilibrium + b),
            "b": b,
            "c": c,
        }
    return parameters, equilibrium


def _truth_rhs_v313(
    mechanism: MechanismV313,
    parameters: dict[str, float],
    value: float,
) -> float:
    x = max(float(value), 1e-12)
    if mechanism == "gompertz_growth":
        return parameters["r"] * x * math.log(parameters["K"] / x)
    if mechanism == "richards_growth":
        return (
            parameters["r"] * x
            * (1.0 - (x / parameters["K"]) ** parameters["nu"])
        )
    return parameters["a"] * x / (parameters["b"] + x) - parameters["c"] * x


def _simulate_truth_v313(
    mechanism: MechanismV313,
    parameters: dict[str, float],
    initial: float,
    times: list[float],
) -> list[float]:
    result = solve_ivp(
        lambda _time, state: [_truth_rhs_v313(mechanism, parameters, state[0])],
        (times[0], times[-1]),
        [initial],
        t_eval=np.asarray(times, dtype=float),
        rtol=1e-10,
        atol=1e-12,
        max_step=0.02,
    )
    if not result.success or result.y.shape[1] != len(times):
        raise RuntimeError("V3.13 truth integration failed")
    return [float(item) for item in result.y[0]]


def generate_growth_worldpacks_v313(
    spec: PrivateGrowthWorldPackSpecV313,
    protocol: PublicGrowthProtocolV313,
    library: CompiledConceptLibraryV313,
    *,
    generated_at: datetime | None = None,
) -> tuple[PublicGrowthWorldPackV313, PrivateGrowthWorldPackV313]:
    for item in (spec, protocol, library):
        item.assert_sealed()
    if (
        spec.public_protocol_hash != protocol.protocol_hash
        or spec.library_hash != library.library_hash
        or protocol.library_hash != library.library_hash
    ):
        raise ValueError("V3.13 worldpack lineage differs")
    times = [
        round(index * protocol.time_step, 12)
        for index in range(protocol.trajectory_points)
    ]
    public_cases: list[PublicGrowthCaseV313] = []
    private_cases: list[PrivateGrowthCaseV313] = []
    for seed_index, seed in enumerate(spec.seeds):
        for mechanism in MECHANISMS_V313:
            parameters, equilibrium = _mechanism_parameters_v313(mechanism, seed)
            pair_id = f"pair_{sha256_value([seed, mechanism])[:16]}"
            for representation in REPRESENTATIONS_V313:
                scale_random = Random(seed + 701 * MECHANISMS_V313.index(mechanism))
                scale = (
                    1.0 if representation == "anonymous_reference"
                    else scale_random.uniform(0.55, 1.85)
                )
                case_id = f"case_{sha256_value([seed, mechanism, representation])[:16]}"
                quality_flags = (
                    ["public_calibration_failure"]
                    if seed_index == spec.calibration_failure_seed_index else []
                )
                public_initials = [
                    equilibrium * fraction for fraction in (0.08, 0.25, 0.68, 1.30)
                ]
                trajectories: list[AnonymousGrowthTrajectoryV313] = []
                noise_random = np.random.default_rng(
                    seed + 1103 * MECHANISMS_V313.index(mechanism)
                    + 17 * REPRESENTATIONS_V313.index(representation)
                )
                for trajectory_index, physical_initial in enumerate(public_initials):
                    physical = np.asarray(
                        _simulate_truth_v313(
                            mechanism, parameters, physical_initial, times
                        ),
                        dtype=float,
                    )
                    observed = physical / scale
                    noise_scale = (
                        spec.observation_noise_fraction
                        * max(float(np.ptp(observed)), float(np.mean(abs(observed))), 1e-6)
                    )
                    noisy = observed + noise_random.normal(
                        0.0, noise_scale, size=observed.shape
                    )
                    noisy = np.maximum(noisy, 1e-8)
                    trajectories.append(AnonymousGrowthTrajectoryV313.seal(
                        trajectory_id=f"trajectory_{case_id}_{trajectory_index}",
                        case_id=case_id,
                        times=times,
                        states=[[float(value)] for value in noisy],
                    ))
                public_case = PublicGrowthCaseV313.seal(
                    case_id=case_id,
                    state_names=["z0"],
                    trajectories=trajectories,
                    quality_flags=quality_flags,
                )
                private_fractions = (0.025, 0.46, 1.80)
                private_initials_physical = [
                    equilibrium * fraction for fraction in private_fractions
                ]
                private_truths = [
                    [
                        [value / scale]
                        for value in _simulate_truth_v313(
                            mechanism, parameters, physical_initial, times
                        )
                    ]
                    for physical_initial in private_initials_physical
                ]
                private_case = PrivateGrowthCaseV313.seal(
                    public_case=public_case,
                    mechanism=mechanism,
                    representation=representation,
                    expected_concept=EXPECTED_CONCEPT_V313[mechanism],
                    hidden_pair_id=pair_id,
                    hidden_seed=seed,
                    physical_parameters=parameters,
                    observed_scale=scale,
                    private_probe_initials=[
                        [value / scale] for value in private_initials_physical
                    ],
                    private_probe_truths=private_truths,
                )
                public_cases.append(public_case)
                private_cases.append(private_case)
    public_pack = PublicGrowthWorldPackV313.seal(
        public_protocol_hash=protocol.protocol_hash,
        library_hash=library.library_hash,
        cases=public_cases,
    )
    private_pack = PrivateGrowthWorldPackV313.seal(
        spec_hash=spec.spec_hash,
        public_pack_hash=public_pack.public_pack_hash,
        cases=private_cases,
        generated_at=generated_at or datetime.now(timezone.utc),
    )
    return public_pack, private_pack


class GrowthModelV313(StrictModel):
    schema_version: Literal["3.13"] = "3.13"
    model_id: Identifier
    case_id: Identifier
    arm: Literal["polynomial_baseline", "evidence_compiled_candidate"]
    concept_id: Identifier
    polynomial_degree: Annotated[int, Field(ge=1, le=4)] | None = None
    polynomial_coefficients: list[Annotated[float, Field(allow_inf_nan=False)]] = Field(
        max_length=5
    )
    package_hash: Sha256 | None = None
    compiled_hash: Sha256 | None = None
    fitted_parameters: dict[Identifier, Annotated[float, Field(allow_inf_nan=False)]]
    parameter_count: Annotated[int, Field(ge=1, le=6)]
    source_trajectory_hashes: list[Sha256] = Field(min_length=2, max_length=2)
    model_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_model(self) -> "GrowthModelV313":
        if self.arm == "polynomial_baseline":
            if (
                self.polynomial_degree is None
                or len(self.polynomial_coefficients) != self.polynomial_degree + 1
                or self.package_hash is not None
                or self.compiled_hash is not None
                or self.fitted_parameters
            ):
                raise ValueError("V3.13 polynomial model payload differs")
        elif (
            self.polynomial_degree is not None
            or self.polynomial_coefficients
            or not self.package_hash
            or not self.compiled_hash
            or len(self.fitted_parameters) != self.parameter_count
        ):
            raise ValueError("V3.13 compiled concept model payload differs")
        if self.model_hash and self.model_hash != self.content_hash():
            raise ValueError("V3.13 growth model hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "model_hash")

    def assert_sealed(self) -> None:
        if not self.model_hash or self.model_hash != self.content_hash():
            raise ValueError("V3.13 growth model is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "GrowthModelV313":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"model_hash"}),
            model_hash=draft.content_hash(),
        )


class GrowthAttemptReceiptV313(StrictModel):
    schema_version: Literal["3.13"] = "3.13"
    attempt_id: Identifier
    case_id: Identifier
    arm: Literal["polynomial_baseline", "evidence_compiled_candidate"]
    concept_id: Identifier
    package_hash: Sha256 | None
    compiled_hash: Sha256 | None
    model: GrowthModelV313 | None
    fit_integral_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    validation_trajectory_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    challenge_trajectory_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    public_score: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    marginal_public_influence: Annotated[float, Field(allow_inf_nan=False)]
    parameter_count: Annotated[int, Field(ge=0, le=6)]
    valid: bool
    public_evaluator_query_count: Literal[1] = 1
    private_values_used: Literal[False] = False
    arbitrary_code_used: Literal[False] = False
    attempt_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_attempt(self) -> "GrowthAttemptReceiptV313":
        if self.arm == "polynomial_baseline" and (
            self.package_hash is not None or self.compiled_hash is not None
        ):
            raise ValueError("V3.13 baseline cannot bind concept package")
        if self.arm == "evidence_compiled_candidate" and (
            self.package_hash is None or self.compiled_hash is None
        ):
            raise ValueError("V3.13 candidate needs compiled package")
        if self.model:
            self.model.assert_sealed()
            if self.model.case_id != self.case_id or self.model.parameter_count != self.parameter_count:
                raise ValueError("V3.13 attempt model binding differs")
        elif self.valid or self.parameter_count:
            raise ValueError("V3.13 missing model cannot be valid")
        if self.attempt_hash and self.attempt_hash != self.content_hash():
            raise ValueError("V3.13 attempt hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "attempt_hash")

    def assert_sealed(self) -> None:
        if not self.attempt_hash or self.attempt_hash != self.content_hash():
            raise ValueError("V3.13 attempt is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "GrowthAttemptReceiptV313":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"attempt_hash"}),
            attempt_hash=draft.content_hash(),
        )


class GrowthDecisionV313(StrictModel):
    schema_version: Literal["3.13"] = "3.13"
    arm: Literal["polynomial_baseline", "evidence_compiled_candidate"]
    selected_attempt_hash: Sha256 | None
    selected_concept_id: Identifier | None
    selected_public_score: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None
    selected_marginal_influence: Annotated[float, Field(allow_inf_nan=False)] | None
    selection_rule: Literal[
        "minimum_public_ood_score_then_complexity_with_candidate_influence_gate"
    ] = "minimum_public_ood_score_then_complexity_with_candidate_influence_gate"
    private_values_used: Literal[False] = False

    @model_validator(mode="after")
    def validate_decision(self) -> "GrowthDecisionV313":
        present = [
            self.selected_attempt_hash is not None,
            self.selected_concept_id is not None,
            self.selected_public_score is not None,
            self.selected_marginal_influence is not None,
        ]
        if any(present) and not all(present):
            raise ValueError("V3.13 decision fields differ")
        return self


class GrowthCaseReceiptV313(StrictModel):
    schema_version: Literal["3.13"] = "3.13"
    case_id: Identifier
    public_case_hash: Sha256
    library_hash: Sha256
    quality_flags: list[Identifier]
    baseline_attempts: list[GrowthAttemptReceiptV313] = Field(max_length=4)
    candidate_attempts: list[GrowthAttemptReceiptV313] = Field(max_length=4)
    baseline_decision: GrowthDecisionV313 | None
    candidate_decision: GrowthDecisionV313 | None
    prediction_arm: Literal["polynomial_baseline", "evidence_compiled_candidate"] | None
    prediction_attempt_hash: Sha256 | None
    prediction_switch_rule: Literal[
        "candidate_only_when_public_score_no_worse_than_baseline"
    ] = "candidate_only_when_public_score_no_worse_than_baseline"
    same_candidate_library_used: Literal[True] = True
    all_attempts_persisted: Literal[True] = True
    private_values_used: Literal[False] = False
    receipt_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_receipt(self) -> "GrowthCaseReceiptV313":
        if self.quality_flags:
            if (
                self.baseline_attempts or self.candidate_attempts
                or self.baseline_decision is not None
                or self.candidate_decision is not None
                or self.prediction_arm is not None
                or self.prediction_attempt_hash is not None
            ):
                raise ValueError("V3.13 quality case must abstain before search")
        elif (
            len(self.baseline_attempts) != 4
            or len(self.candidate_attempts) != 4
            or self.baseline_decision is None
            or self.candidate_decision is None
            or self.prediction_arm is None
            or self.prediction_attempt_hash is None
        ):
            raise ValueError("V3.13 performance receipt search matrix differs")
        elif self.prediction_attempt_hash not in {
            item.attempt_hash for item in self.baseline_attempts + self.candidate_attempts
        }:
            raise ValueError("V3.13 prediction attempt binding differs")
        elif not any(
            item.attempt_hash == self.prediction_attempt_hash
            and item.arm == self.prediction_arm
            for item in self.baseline_attempts + self.candidate_attempts
        ):
            raise ValueError("V3.13 prediction arm binding differs")
        for item in self.baseline_attempts + self.candidate_attempts:
            item.assert_sealed()
        if self.receipt_hash and self.receipt_hash != self.content_hash():
            raise ValueError("V3.13 case receipt hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "receipt_hash")

    def assert_sealed(self) -> None:
        if not self.receipt_hash or self.receipt_hash != self.content_hash():
            raise ValueError("V3.13 case receipt is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "GrowthCaseReceiptV313":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"receipt_hash"}),
            receipt_hash=draft.content_hash(),
        )


class GrowthDiscoveryBundleV313(StrictModel):
    schema_version: Literal["3.13"] = "3.13"
    bundle_id: Identifier
    public_protocol_hash: Sha256
    public_pack_hash: Sha256
    policy_hash: Sha256
    library_hash: Sha256
    case_receipts: list[GrowthCaseReceiptV313] = Field(min_length=36, max_length=48)
    created_at: datetime
    bundle_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_bundle(self) -> "GrowthDiscoveryBundleV313":
        _assert_timezone(self.created_at, "created_at")
        ids = [item.case_id for item in self.case_receipts]
        if len(ids) != len(set(ids)):
            raise ValueError("V3.13 bundle case ids differ")
        for item in self.case_receipts:
            item.assert_sealed()
        if self.bundle_hash and self.bundle_hash != self.content_hash():
            raise ValueError("V3.13 bundle hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "bundle_hash")

    def assert_sealed(self) -> None:
        if not self.bundle_hash or self.bundle_hash != self.content_hash():
            raise ValueError("V3.13 bundle is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "GrowthDiscoveryBundleV313":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"bundle_hash"}),
            bundle_hash=draft.content_hash(),
        )


def _case_arrays_v313(
    case: PublicGrowthCaseV313,
    protocol: PublicGrowthProtocolV313,
) -> tuple[np.ndarray, np.ndarray]:
    states = []
    derivatives = []
    for trajectory in case.trajectories[: protocol.fit_trajectory_count]:
        values = np.asarray([row[0] for row in trajectory.states], dtype=float)
        derivative = savgol_filter(
            values,
            protocol.savgol_window,
            protocol.savgol_polynomial,
            deriv=1,
            delta=protocol.time_step,
            mode="interp",
        )
        states.append(values)
        derivatives.append(derivative)
    return np.concatenate(states), np.concatenate(derivatives)


def _integral_targets_v313(
    case: PublicGrowthCaseV313,
    protocol: PublicGrowthProtocolV313,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    targets = []
    for trajectory in case.trajectories[: protocol.fit_trajectory_count]:
        states = np.asarray([row[0] for row in trajectory.states], dtype=float)
        delta = states[1:] - states[0]
        scale = max(float(np.ptp(states)), float(np.mean(np.abs(states))), 1e-6)
        targets.append((states, delta, scale))
    return targets


def _cumulative_trapezoid_v313(values: np.ndarray, step: float) -> np.ndarray:
    return np.cumsum((values[:-1] + values[1:]) * 0.5 * step)


def _model_rhs_v313(
    model: GrowthModelV313,
    value: float,
    library: CompiledConceptLibraryV313,
) -> float:
    if model.arm == "polynomial_baseline":
        return float(sum(
            coefficient * value ** power
            for power, coefficient in enumerate(model.polynomial_coefficients)
        ))
    entry = next(
        item for item in library.entries
        if item.package.package_hash == model.package_hash
    )
    return evaluate_operator_ast_v313(
        entry.package.rhs, [value], model.fitted_parameters
    )


def _simulate_model_v313(
    model: GrowthModelV313,
    initial: float,
    times: list[float],
    library: CompiledConceptLibraryV313,
) -> list[list[float]]:
    result = solve_ivp(
        lambda _time, state: [_model_rhs_v313(model, state[0], library)],
        (times[0], times[-1]),
        [initial],
        t_eval=np.asarray(times, dtype=float),
        rtol=1e-7,
        atol=1e-9,
        max_step=0.03,
    )
    if (
        not result.success
        or result.y.shape[1] != len(times)
        or not np.all(np.isfinite(result.y))
        or np.max(np.abs(result.y)) > 1e6
    ):
        raise RuntimeError("V3.13 candidate integration failed")
    return [[float(value)] for value in result.y[0]]


def _trajectory_loss_v313(
    case: PublicGrowthCaseV313,
    trajectory_index: int,
    model: GrowthModelV313,
    library: CompiledConceptLibraryV313,
) -> float:
    trajectory = case.trajectories[trajectory_index]
    prediction = _simulate_model_v313(
        model, trajectory.states[0][0], trajectory.times, library
    )
    return trajectory_nrmse(trajectory.states, prediction)


def _fit_polynomial_v313(
    case: PublicGrowthCaseV313,
    degree: int,
    protocol: PublicGrowthProtocolV313,
) -> tuple[GrowthModelV313, float]:
    design_blocks = []
    target_blocks = []
    for states, delta, scale in _integral_targets_v313(case, protocol):
        design_blocks.append(np.column_stack([
            _cumulative_trapezoid_v313(states ** power, protocol.time_step) / scale
            for power in range(degree + 1)
        ]))
        target_blocks.append(delta / scale)
    design = np.vstack(design_blocks)
    target = np.concatenate(target_blocks)
    matrix = design.T @ design + protocol.ridge_alpha * np.eye(degree + 1)
    coefficients = np.linalg.solve(matrix, design.T @ target)
    prediction = design @ coefficients
    fit_loss = float(np.sqrt(np.mean((prediction - target) ** 2)))
    return GrowthModelV313.seal(
        model_id=f"model_{case.case_id}_polynomial_{degree}",
        case_id=case.case_id,
        arm="polynomial_baseline",
        concept_id=f"polynomial_degree_{degree}",
        polynomial_degree=degree,
        polynomial_coefficients=[float(item) for item in coefficients],
        package_hash=None,
        compiled_hash=None,
        fitted_parameters={},
        parameter_count=degree + 1,
        source_trajectory_hashes=[
            item.trajectory_hash
            for item in case.trajectories[: protocol.fit_trajectory_count]
        ],
    ), fit_loss


def _fit_compiled_v313(
    case: PublicGrowthCaseV313,
    entry: CompiledConceptEntryV313,
    protocol: PublicGrowthProtocolV313,
) -> tuple[GrowthModelV313, float]:
    integral_targets = _integral_targets_v313(case, protocol)
    parameter_specs = entry.package.parameters

    def residual(values: np.ndarray) -> np.ndarray:
        parameters = {
            item.parameter_id: float(value)
            for item, value in zip(parameter_specs, values, strict=True)
        }
        blocks = []
        for states, delta, scale in integral_targets:
            predictions = []
            for state in states:
                try:
                    prediction = evaluate_operator_ast_v313(
                        entry.package.rhs, [float(state)], parameters
                    )
                    if not math.isfinite(prediction) or abs(prediction) > 1e6:
                        return np.full(sum(len(item[1]) for item in integral_targets), 1e6)
                    predictions.append(prediction)
                except (ValueError, OverflowError, ZeroDivisionError):
                    return np.full(sum(len(item[1]) for item in integral_targets), 1e6)
            integrated = _cumulative_trapezoid_v313(
                np.asarray(predictions), protocol.time_step
            )
            blocks.append((integrated - delta) / scale)
        return np.concatenate(blocks)

    lower = np.asarray([item.lower_bound for item in parameter_specs])
    upper = np.asarray([item.upper_bound for item in parameter_specs])
    starts = [
        np.asarray([item.initial_value for item in parameter_specs]),
        lower + 0.25 * (upper - lower),
        lower + 0.50 * (upper - lower),
        lower + 0.75 * (upper - lower),
    ]
    results = [
        least_squares(
            residual,
            x0=start,
            bounds=(lower, upper),
            max_nfev=800,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
        )
        for start in starts
    ]
    viable = [
        item for item in results
        if item.success and np.all(np.isfinite(item.x))
    ]
    if not viable:
        raise RuntimeError("V3.13 compiled parameter optimization failed")
    result = min(viable, key=lambda item: float(np.mean(residual(item.x) ** 2)))
    parameters = {
        item.parameter_id: float(value)
        for item, value in zip(parameter_specs, result.x, strict=True)
    }
    fit_loss = float(np.sqrt(np.mean(residual(result.x) ** 2)))
    return GrowthModelV313.seal(
        model_id=f"model_{case.case_id}_{entry.package.concept_id}",
        case_id=case.case_id,
        arm="evidence_compiled_candidate",
        concept_id=entry.package.concept_id,
        polynomial_degree=None,
        polynomial_coefficients=[],
        package_hash=entry.package.package_hash,
        compiled_hash=entry.compiled.compiled_hash,
        fitted_parameters=parameters,
        parameter_count=len(parameters),
        source_trajectory_hashes=[
            item.trajectory_hash
            for item in case.trajectories[: protocol.fit_trajectory_count]
        ],
    ), fit_loss


def _attempt_payload_v313(
    case: PublicGrowthCaseV313,
    *,
    arm: Literal["polynomial_baseline", "evidence_compiled_candidate"],
    concept_id: str,
    package_hash: str | None,
    compiled_hash: str | None,
    model: GrowthModelV313 | None,
    fit_loss: float,
    protocol: PublicGrowthProtocolV313,
    library: CompiledConceptLibraryV313,
) -> dict[str, object]:
    if model is None:
        return {
            "model": None,
            "fit_integral_loss": protocol.unresolved_loss,
            "validation_trajectory_loss": protocol.unresolved_loss,
            "challenge_trajectory_loss": protocol.unresolved_loss,
            "public_score": protocol.unresolved_loss,
            "parameter_count": 0,
            "valid": False,
        }
    validation = _trajectory_loss_v313(
        case, protocol.validation_trajectory_index, model, library
    )
    challenge = _trajectory_loss_v313(
        case, protocol.challenge_trajectory_index, model, library
    )
    score = max(validation, challenge) + protocol.complexity_penalty * model.parameter_count
    return {
        "model": model,
        "fit_integral_loss": fit_loss,
        "validation_trajectory_loss": validation,
        "challenge_trajectory_loss": challenge,
        "public_score": score,
        "parameter_count": model.parameter_count,
        "valid": bool(
            np.isfinite(score)
            and (
                arm == "polynomial_baseline"
                or score <= protocol.maximum_candidate_public_score
            )
        ),
    }


def _select_growth_attempt_v313(
    arm: Literal["polynomial_baseline", "evidence_compiled_candidate"],
    attempts: list[GrowthAttemptReceiptV313],
    protocol: PublicGrowthProtocolV313,
) -> GrowthDecisionV313:
    eligible = (
        [item for item in attempts if item.model is not None]
        if arm == "polynomial_baseline"
        else [
            item for item in attempts
            if item.valid
            and item.marginal_public_influence
            >= protocol.minimum_selected_concept_influence
        ]
    )
    if not eligible:
        return GrowthDecisionV313(
            arm=arm,
            selected_attempt_hash=None,
            selected_concept_id=None,
            selected_public_score=None,
            selected_marginal_influence=None,
        )
    selected = min(
        eligible,
        key=lambda item: (
            item.public_score,
            item.parameter_count,
            item.concept_id,
        ),
    )
    return GrowthDecisionV313(
        arm=arm,
        selected_attempt_hash=selected.attempt_hash,
        selected_concept_id=selected.concept_id,
        selected_public_score=selected.public_score,
        selected_marginal_influence=selected.marginal_public_influence,
    )


def _execute_growth_case_v313(
    case: PublicGrowthCaseV313,
    protocol: PublicGrowthProtocolV313,
    library: CompiledConceptLibraryV313,
) -> GrowthCaseReceiptV313:
    case.assert_sealed()
    if case.quality_flags:
        return GrowthCaseReceiptV313.seal(
            case_id=case.case_id,
            public_case_hash=case.public_hash,
            library_hash=library.library_hash,
            quality_flags=case.quality_flags,
            baseline_attempts=[],
            candidate_attempts=[],
            baseline_decision=None,
            candidate_decision=None,
            prediction_arm=None,
            prediction_attempt_hash=None,
        )
    baseline_payloads = []
    for degree in BASELINE_DEGREES_V313:
        try:
            model, fit_loss = _fit_polynomial_v313(case, degree, protocol)
            payload = _attempt_payload_v313(
                case,
                arm="polynomial_baseline",
                concept_id=f"polynomial_degree_{degree}",
                package_hash=None,
                compiled_hash=None,
                model=model,
                fit_loss=fit_loss,
                protocol=protocol,
                library=library,
            )
        except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError):
            payload = _attempt_payload_v313(
                case,
                arm="polynomial_baseline",
                concept_id=f"polynomial_degree_{degree}",
                package_hash=None,
                compiled_hash=None,
                model=None,
                fit_loss=protocol.unresolved_loss,
                protocol=protocol,
                library=library,
            )
        baseline_payloads.append((degree, payload))
    baseline_attempts = [
        GrowthAttemptReceiptV313.seal(
            attempt_id=f"attempt_{case.case_id}_baseline_{degree}",
            case_id=case.case_id,
            arm="polynomial_baseline",
            concept_id=f"polynomial_degree_{degree}",
            package_hash=None,
            compiled_hash=None,
            marginal_public_influence=0.0,
            **payload,
        )
        for degree, payload in baseline_payloads
    ]
    candidate_payloads = []
    for entry in library.entries:
        try:
            model, fit_loss = _fit_compiled_v313(case, entry, protocol)
            payload = _attempt_payload_v313(
                case,
                arm="evidence_compiled_candidate",
                concept_id=entry.package.concept_id,
                package_hash=entry.package.package_hash,
                compiled_hash=entry.compiled.compiled_hash,
                model=model,
                fit_loss=fit_loss,
                protocol=protocol,
                library=library,
            )
        except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError):
            payload = _attempt_payload_v313(
                case,
                arm="evidence_compiled_candidate",
                concept_id=entry.package.concept_id,
                package_hash=entry.package.package_hash,
                compiled_hash=entry.compiled.compiled_hash,
                model=None,
                fit_loss=protocol.unresolved_loss,
                protocol=protocol,
                library=library,
            )
        candidate_payloads.append((entry, payload))
    scores = [float(payload["public_score"]) for _, payload in candidate_payloads]
    candidate_attempts = []
    for index, (entry, payload) in enumerate(candidate_payloads):
        best_other = min(score for other, score in enumerate(scores) if other != index)
        influence = best_other - scores[index]
        candidate_attempts.append(GrowthAttemptReceiptV313.seal(
            attempt_id=f"attempt_{case.case_id}_candidate_{index + 1}",
            case_id=case.case_id,
            arm="evidence_compiled_candidate",
            concept_id=entry.package.concept_id,
            package_hash=entry.package.package_hash,
            compiled_hash=entry.compiled.compiled_hash,
            marginal_public_influence=influence,
            **payload,
        ))
    baseline_decision = _select_growth_attempt_v313(
        "polynomial_baseline", baseline_attempts, protocol
    )
    candidate_decision = _select_growth_attempt_v313(
        "evidence_compiled_candidate", candidate_attempts, protocol
    )
    if baseline_decision.selected_attempt_hash is None:
        if candidate_decision.selected_attempt_hash is None:
            raise ValueError("V3.13 performance case has no fitted prediction")
        prediction_arm = "evidence_compiled_candidate"
        prediction_attempt_hash = candidate_decision.selected_attempt_hash
    elif (
        candidate_decision.selected_attempt_hash is not None
        and candidate_decision.selected_public_score
        <= baseline_decision.selected_public_score
    ):
        prediction_arm = "evidence_compiled_candidate"
        prediction_attempt_hash = candidate_decision.selected_attempt_hash
    else:
        prediction_arm = "polynomial_baseline"
        prediction_attempt_hash = baseline_decision.selected_attempt_hash
    return GrowthCaseReceiptV313.seal(
        case_id=case.case_id,
        public_case_hash=case.public_hash,
        library_hash=library.library_hash,
        quality_flags=case.quality_flags,
        baseline_attempts=baseline_attempts,
        candidate_attempts=candidate_attempts,
        baseline_decision=baseline_decision,
        candidate_decision=candidate_decision,
        prediction_arm=prediction_arm,
        prediction_attempt_hash=prediction_attempt_hash,
    )


def execute_evidence_compiled_growth_v313(
    public_protocol: PublicGrowthProtocolV313,
    public_pack: PublicGrowthWorldPackV313,
    policy: EvidenceConceptPolicyV313,
    library: CompiledConceptLibraryV313,
    *,
    executed_at: datetime,
) -> GrowthDiscoveryBundleV313:
    for item in (public_protocol, public_pack, policy, library):
        item.assert_sealed()
    if (
        public_pack.public_protocol_hash != public_protocol.protocol_hash
        or public_pack.library_hash != library.library_hash
        or policy.library_hash != library.library_hash
        or public_protocol.policy_hash != policy.policy_hash
        or policy.candidate_package_hashes
        != [item.package.package_hash for item in library.entries]
    ):
        raise ValueError("V3.13 public execution binding differs")
    return GrowthDiscoveryBundleV313.seal(
        bundle_id="evidence_compiled_growth_bundle_v313",
        public_protocol_hash=public_protocol.protocol_hash,
        public_pack_hash=public_pack.public_pack_hash,
        policy_hash=policy.policy_hash,
        library_hash=library.library_hash,
        case_receipts=[
            _execute_growth_case_v313(case, public_protocol, library)
            for case in public_pack.cases
        ],
        created_at=executed_at,
    )


class PrivateGrowthCaseResultV313(StrictModel):
    schema_version: Literal["3.13"] = "3.13"
    case_id: Identifier
    hidden_pair_id: Identifier
    mechanism: MechanismV313
    representation: RepresentationV313
    expected_concept: Identifier
    baseline_concept: Identifier | None
    candidate_concept: Identifier | None
    selected_marginal_influence: Annotated[float, Field(allow_inf_nan=False)] | None
    prediction_arm: Literal["polynomial_baseline", "evidence_compiled_candidate"]
    baseline_target_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    concept_model_target_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    candidate_target_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    candidate_improvement: Annotated[float, Field(allow_inf_nan=False)]
    concept_correct: bool
    material_negative_transfer: bool

    @model_validator(mode="after")
    def validate_result(self) -> "PrivateGrowthCaseResultV313":
        if not math.isclose(
            self.candidate_improvement,
            self.baseline_target_loss - self.candidate_target_loss,
            abs_tol=1e-12,
        ):
            raise ValueError("V3.13 case improvement differs")
        return self


class GrowthRepresentationPairResultV313(StrictModel):
    schema_version: Literal["3.13"] = "3.13"
    hidden_pair_id: Identifier
    mechanism: MechanismV313
    reference_case_id: Identifier
    scaled_case_id: Identifier
    reference_concept: Identifier | None
    scaled_concept: Identifier | None
    concept_consistent: bool
    reference_target_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    scaled_target_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    absolute_target_loss_difference: Annotated[float, Field(ge=0, allow_inf_nan=False)]


class ConceptAdjudicationEntryV313(StrictModel):
    concept_id: Identifier
    package_hash: Sha256
    compiled_hash: Sha256
    status: Literal[
        "development_candidate",
        "privately_admitted",
        "privately_contradicted",
    ]
    expected_case_count: Annotated[int, Field(ge=0)]
    selected_case_count: Annotated[int, Field(ge=0)]
    correct_case_count: Annotated[int, Field(ge=0)]
    private_mean_improvement: Annotated[float, Field(allow_inf_nan=False)]
    public_score_used_for_admission: Literal[False] = False
    generator_receives_private_feedback: Literal[False] = False


class PrivateConceptAdjudicationV313(StrictModel):
    schema_version: Literal["3.13"] = "3.13"
    adjudication_id: Identifier
    phase: Literal["development", "confirmation"]
    private_spec_hash: Sha256
    private_pack_hash: Sha256
    bundle_hash: Sha256
    entries: list[ConceptAdjudicationEntryV313] = Field(min_length=4, max_length=4)
    private_evaluator_only: Literal[True] = True
    returned_to_generator: Literal[False] = False
    created_at: datetime
    adjudication_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_adjudication(self) -> "PrivateConceptAdjudicationV313":
        _assert_timezone(self.created_at, "created_at")
        if self.adjudication_hash and self.adjudication_hash != self.content_hash():
            raise ValueError("V3.13 adjudication hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "adjudication_hash")

    def assert_sealed(self) -> None:
        if not self.adjudication_hash or self.adjudication_hash != self.content_hash():
            raise ValueError("V3.13 adjudication is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "PrivateConceptAdjudicationV313":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"adjudication_hash"}),
            adjudication_hash=draft.content_hash(),
        )


class EvidenceCompiledGrowthReportV313(StrictModel):
    schema_version: Literal["3.13"] = "3.13"
    report_id: Identifier
    phase: Literal["development", "confirmation"]
    private_spec_hash: Sha256
    public_protocol_hash: Sha256
    public_pack_hash: Sha256
    private_pack_hash: Sha256
    bundle_hash: Sha256
    library_hash: Sha256
    lineage_receipt_hash: Sha256
    source_v312_report_hash: Sha256
    adjudication_hash: Sha256
    input_experience_store_hash: Sha256 | None
    output_experience_store_hash: Sha256
    case_results: list[PrivateGrowthCaseResultV313]
    pair_results: list[GrowthRepresentationPairResultV313]
    performance_case_count: Annotated[int, Field(ge=1)]
    quality_case_count: Annotated[int, Field(ge=0)]
    baseline_expression_evaluation_count: Annotated[int, Field(ge=1)]
    candidate_expression_evaluation_count: Annotated[int, Field(ge=1)]
    baseline_coverage: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    candidate_coverage: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    baseline_mean_target_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    concept_model_mean_target_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    candidate_mean_target_loss: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    paired_mean_improvement: Annotated[float, Field(allow_inf_nan=False)]
    paired_improvement_ci_lower: Annotated[float, Field(allow_inf_nan=False)]
    paired_improvement_ci_upper: Annotated[float, Field(allow_inf_nan=False)]
    concept_recovery_accuracy: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    pair_concept_consistency: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    maximum_pair_loss_difference: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    selected_positive_influence_rate: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    candidate_prediction_use_rate: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    mean_loss_by_mechanism_baseline: dict[MechanismV313, Annotated[float, Field(ge=0, allow_inf_nan=False)]]
    mean_loss_by_mechanism_candidate: dict[MechanismV313, Annotated[float, Field(ge=0, allow_inf_nan=False)]]
    mean_loss_by_representation_candidate: dict[RepresentationV313, Annotated[float, Field(ge=0, allow_inf_nan=False)]]
    candidate_selection_counts: dict[Identifier, Annotated[int, Field(ge=0)]]
    material_negative_transfer_count: Annotated[int, Field(ge=0)]
    material_negative_transfer_upper_95: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    public_execution_private_blind: bool
    all_attempts_persisted: bool
    equal_evaluation_budget: bool
    gates: dict[Identifier, bool]
    ready_for_concept_admission: bool
    task_router_permitted: Literal[False] = False
    model_qualification_permitted: Literal[False] = False
    real_world_execution_permitted: Literal[False] = False
    status: Literal[
        "evidence_concept_development_diagnostic_v313",
        "evidence_compiled_concepts_admitted_v313",
        "evidence_compiled_concepts_refuted_v313",
    ]
    created_at: datetime
    report_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_report(self) -> "EvidenceCompiledGrowthReportV313":
        _assert_timezone(self.created_at, "created_at")
        expected_ready = self.phase == "confirmation" and all(self.gates.values())
        expected_status = (
            "evidence_concept_development_diagnostic_v313"
            if self.phase == "development"
            else (
                "evidence_compiled_concepts_admitted_v313"
                if expected_ready
                else "evidence_compiled_concepts_refuted_v313"
            )
        )
        if self.ready_for_concept_admission != expected_ready or self.status != expected_status:
            raise ValueError("V3.13 report status differs")
        if self.report_hash and self.report_hash != self.content_hash():
            raise ValueError("V3.13 report hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "report_hash")

    def assert_sealed(self) -> None:
        if not self.report_hash or self.report_hash != self.content_hash():
            raise ValueError("V3.13 report is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "EvidenceCompiledGrowthReportV313":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"report_hash"}),
            report_hash=draft.content_hash(),
        )


@dataclass(frozen=True)
class GrowthEvaluationV313:
    adjudication: PrivateConceptAdjudicationV313
    experience_store: ConceptExperienceStoreV313
    report: EvidenceCompiledGrowthReportV313


def _selected_attempt_v313(
    receipt: GrowthCaseReceiptV313,
    decision: GrowthDecisionV313,
) -> GrowthAttemptReceiptV313 | None:
    if decision.selected_attempt_hash is None:
        return None
    attempts = (
        receipt.baseline_attempts
        if decision.arm == "polynomial_baseline" else receipt.candidate_attempts
    )
    matches = [
        item for item in attempts
        if item.attempt_hash == decision.selected_attempt_hash
    ]
    if len(matches) != 1:
        raise ValueError("V3.13 selected attempt binding differs")
    return matches[0]


def _private_growth_loss_v313(
    private_case: PrivateGrowthCaseV313,
    attempt: GrowthAttemptReceiptV313 | None,
    protocol: PublicGrowthProtocolV313,
    library: CompiledConceptLibraryV313,
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
            prediction = _simulate_model_v313(
                attempt.model, initial[0], times, library
            )
            losses.append(trajectory_nrmse(truth, prediction))
        except (RuntimeError, ValueError, OverflowError):
            losses.append(protocol.unresolved_loss)
    return float(np.mean(losses))


def _bootstrap_ci_v313(
    values: np.ndarray,
    spec: PrivateGrowthWorldPackSpecV313,
) -> tuple[float, float]:
    random = np.random.default_rng(spec.bootstrap_seed)
    indices = random.integers(
        0, len(values), size=(spec.bootstrap_replicates, len(values))
    )
    means = np.mean(values[indices], axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _initial_experience_store_v313(
    evidence: ConceptEvidenceBundleV313,
    library: CompiledConceptLibraryV313,
    *,
    created_at: datetime,
) -> ConceptExperienceStoreV313:
    store = empty_concept_experience_store_v313(evidence, created_at=created_at)
    for entry in library.entries:
        store = append_concept_experience_event_v313(
            store,
            event_type="proposed",
            package=entry.package,
            compiled=None,
            phase="research",
            created_at=created_at,
        )
        store = append_concept_experience_event_v313(
            store,
            event_type="compiled",
            package=entry.package,
            compiled=entry.compiled,
            phase="research",
            created_at=created_at,
        )
    return store


def evaluate_evidence_compiled_growth_v313(
    private_spec: PrivateGrowthWorldPackSpecV313,
    public_protocol: PublicGrowthProtocolV313,
    lineage: VerifiedV312LineageReceiptV313,
    evidence: ConceptEvidenceBundleV313,
    library: CompiledConceptLibraryV313,
    private_pack: PrivateGrowthWorldPackV313,
    bundle: GrowthDiscoveryBundleV313,
    *,
    evaluated_at: datetime,
    input_experience_store: ConceptExperienceStoreV313 | None = None,
) -> GrowthEvaluationV313:
    for item in (
        private_spec, public_protocol, lineage, evidence, library,
        private_pack, bundle,
    ):
        item.assert_sealed()
    if (
        private_pack.spec_hash != private_spec.spec_hash
        or private_pack.public_pack_hash != bundle.public_pack_hash
        or bundle.public_protocol_hash != public_protocol.protocol_hash
        or bundle.library_hash != library.library_hash
        or evidence.evidence_hash != library.evidence_hash
        or lineage.receipt_hash != private_spec.lineage_receipt_hash
    ):
        raise ValueError("V3.13 evaluator binding differs")
    if private_spec.phase == "development":
        if input_experience_store is not None:
            raise ValueError("V3.13 development starts a new experience store")
        experience = _initial_experience_store_v313(
            evidence, library, created_at=public_protocol.frozen_at
        )
        input_store_hash = None
    else:
        if input_experience_store is None:
            raise ValueError("V3.13 confirmation needs development experience")
        input_experience_store.assert_sealed()
        if (
            input_experience_store.store_hash
            != private_spec.development_experience_store_hash
            or input_experience_store.evidence_hash != evidence.evidence_hash
        ):
            raise ValueError("V3.13 development experience binding differs")
        experience = input_experience_store
        input_store_hash = input_experience_store.store_hash
    receipts = {item.case_id: item for item in bundle.case_receipts}
    results: list[PrivateGrowthCaseResultV313] = []
    quality_count = 0
    baseline_evaluations = 0
    candidate_evaluations = 0
    all_persisted = True
    private_blind = True
    selection_counts: dict[str, int] = defaultdict(int)
    baseline_by_mechanism: dict[str, list[float]] = defaultdict(list)
    candidate_by_mechanism: dict[str, list[float]] = defaultdict(list)
    candidate_by_representation: dict[str, list[float]] = defaultdict(list)
    for private_case in private_pack.cases:
        receipt = receipts[private_case.public_case.case_id]
        if receipt.quality_flags:
            quality_count += 1
            continue
        baseline_evaluations += sum(
            item.public_evaluator_query_count for item in receipt.baseline_attempts
        )
        candidate_evaluations += sum(
            item.public_evaluator_query_count for item in receipt.candidate_attempts
        )
        attempts = receipt.baseline_attempts + receipt.candidate_attempts
        all_persisted = all_persisted and (
            len(attempts) == 8
            and len({item.attempt_hash for item in attempts}) == 8
            and receipt.all_attempts_persisted
        )
        private_blind = private_blind and (
            not receipt.private_values_used
            and receipt.same_candidate_library_used
            and all(
                not item.private_values_used and not item.arbitrary_code_used
                for item in attempts
            )
        )
        baseline_attempt = _selected_attempt_v313(
            receipt, receipt.baseline_decision
        )
        candidate_attempt = _selected_attempt_v313(
            receipt, receipt.candidate_decision
        )
        prediction_attempt = next(
            item for item in attempts
            if item.attempt_hash == receipt.prediction_attempt_hash
        )
        baseline_loss = _private_growth_loss_v313(
            private_case, baseline_attempt, public_protocol, library
        )
        candidate_loss = _private_growth_loss_v313(
            private_case, candidate_attempt, public_protocol, library
        )
        safeguarded_loss = _private_growth_loss_v313(
            private_case, prediction_attempt, public_protocol, library
        )
        candidate_concept = receipt.candidate_decision.selected_concept_id
        if candidate_concept:
            selection_counts[candidate_concept] += 1
        baseline_by_mechanism[private_case.mechanism].append(baseline_loss)
        candidate_by_mechanism[private_case.mechanism].append(safeguarded_loss)
        candidate_by_representation[private_case.representation].append(safeguarded_loss)
        results.append(PrivateGrowthCaseResultV313(
            case_id=private_case.public_case.case_id,
            hidden_pair_id=private_case.hidden_pair_id,
            mechanism=private_case.mechanism,
            representation=private_case.representation,
            expected_concept=private_case.expected_concept,
            baseline_concept=receipt.baseline_decision.selected_concept_id,
            candidate_concept=candidate_concept,
            selected_marginal_influence=(
                receipt.candidate_decision.selected_marginal_influence
            ),
            prediction_arm=receipt.prediction_arm,
            baseline_target_loss=baseline_loss,
            concept_model_target_loss=candidate_loss,
            candidate_target_loss=safeguarded_loss,
            candidate_improvement=baseline_loss - safeguarded_loss,
            concept_correct=(candidate_concept == private_case.expected_concept),
            material_negative_transfer=(
                safeguarded_loss - baseline_loss
                > private_spec.material_negative_transfer
            ),
        ))
    by_pair: dict[str, list[PrivateGrowthCaseResultV313]] = defaultdict(list)
    for item in results:
        by_pair[item.hidden_pair_id].append(item)
    pairs: list[GrowthRepresentationPairResultV313] = []
    for pair_id, members in by_pair.items():
        if len(members) != 2:
            raise ValueError("V3.13 representation pair incomplete")
        reference = next(
            item for item in members if item.representation == "anonymous_reference"
        )
        scaled = next(
            item for item in members if item.representation == "anonymous_scaled"
        )
        pairs.append(GrowthRepresentationPairResultV313(
            hidden_pair_id=pair_id,
            mechanism=reference.mechanism,
            reference_case_id=reference.case_id,
            scaled_case_id=scaled.case_id,
            reference_concept=reference.candidate_concept,
            scaled_concept=scaled.candidate_concept,
            concept_consistent=(reference.candidate_concept == scaled.candidate_concept),
            reference_target_loss=reference.candidate_target_loss,
            scaled_target_loss=scaled.candidate_target_loss,
            absolute_target_loss_difference=abs(
                reference.candidate_target_loss - scaled.candidate_target_loss
            ),
        ))
    baseline_losses = np.asarray([item.baseline_target_loss for item in results])
    concept_model_losses = np.asarray([
        item.concept_model_target_loss for item in results
    ])
    candidate_losses = np.asarray([item.candidate_target_loss for item in results])
    improvements = baseline_losses - candidate_losses
    ci_lower, ci_upper = _bootstrap_ci_v313(improvements, private_spec)
    negatives = sum(item.material_negative_transfer for item in results)
    negative_upper = (
        float(beta.ppf(0.95, negatives + 1, len(results) - negatives))
        if len(results) > negatives else 1.0
    )
    mechanism_baseline = {
        mechanism: float(np.mean(baseline_by_mechanism[mechanism]))
        for mechanism in MECHANISMS_V313
    }
    mechanism_candidate = {
        mechanism: float(np.mean(candidate_by_mechanism[mechanism]))
        for mechanism in MECHANISMS_V313
    }
    representation_candidate = {
        representation: float(np.mean(candidate_by_representation[representation]))
        for representation in REPRESENTATIONS_V313
    }
    recovery_accuracy = float(np.mean([item.concept_correct for item in results]))
    pair_consistency = float(np.mean([item.concept_consistent for item in pairs]))
    max_pair_difference = max(item.absolute_target_loss_difference for item in pairs)
    positive_influence_rate = float(np.mean([
        item.selected_marginal_influence is not None
        and item.selected_marginal_influence
        >= public_protocol.minimum_selected_concept_influence
        for item in results
    ]))
    entries: list[ConceptAdjudicationEntryV313] = []
    for library_entry in library.entries:
        concept_id = library_entry.package.concept_id
        relevant = [item for item in results if item.expected_concept == concept_id]
        selected = [item for item in results if item.candidate_concept == concept_id]
        correct = sum(item.concept_correct for item in relevant)
        mean_improvement = (
            float(np.mean([
                item.baseline_target_loss - item.concept_model_target_loss
                for item in relevant
            ]))
            if relevant else 0.0
        )
        eligible = bool(
            relevant
            and correct / len(relevant)
            >= private_spec.minimum_concept_recovery_accuracy
            and mean_improvement >= -private_spec.maximum_mechanism_regression
        )
        if private_spec.phase == "development":
            status = "development_candidate"
        elif eligible:
            status = "privately_admitted"
        else:
            status = "privately_contradicted"
        entries.append(ConceptAdjudicationEntryV313(
            concept_id=concept_id,
            package_hash=library_entry.package.package_hash,
            compiled_hash=library_entry.compiled.compiled_hash,
            status=status,
            expected_case_count=len(relevant),
            selected_case_count=len(selected),
            correct_case_count=correct,
            private_mean_improvement=mean_improvement,
        ))
    adjudication = PrivateConceptAdjudicationV313.seal(
        adjudication_id=f"adjudication_{private_spec.experiment_id}",
        phase=private_spec.phase,
        private_spec_hash=private_spec.spec_hash,
        private_pack_hash=private_pack.pack_hash,
        bundle_hash=bundle.bundle_hash,
        entries=entries,
        created_at=evaluated_at,
    )
    for entry, library_entry in zip(entries, library.entries, strict=True):
        if private_spec.phase == "development":
            if entry.expected_case_count and entry.correct_case_count == entry.expected_case_count:
                experience = append_concept_experience_event_v313(
                    experience,
                    event_type="development_supported",
                    package=library_entry.package,
                    compiled=library_entry.compiled,
                    phase="development",
                    created_at=evaluated_at,
                    attempt_hash=bundle.bundle_hash,
                )
        elif entry.status == "privately_admitted":
            experience = append_concept_experience_event_v313(
                experience,
                event_type="privately_admitted",
                package=library_entry.package,
                compiled=library_entry.compiled,
                phase="confirmation",
                created_at=evaluated_at,
                adjudication_hash=adjudication.adjudication_hash,
                private_evaluator_event=True,
            )
        else:
            experience = append_concept_experience_event_v313(
                experience,
                event_type="contradicted",
                package=library_entry.package,
                compiled=library_entry.compiled,
                phase="confirmation",
                created_at=evaluated_at,
                adjudication_hash=adjudication.adjudication_hash,
                private_evaluator_event=True,
            )
    equal_budget = (
        baseline_evaluations == candidate_evaluations == len(results) * 4
    )
    expected_active = (
        {} if private_spec.phase == "development"
        else {
            "log_capacity_growth": 1,
            "generalized_capacity_growth": 1,
            "hyperbolic_net_growth": 1,
        }
    )
    expected_statuses = (
        all(item.status == "development_candidate" for item in entries)
        if private_spec.phase == "development"
        else (
            all(item.status == "privately_admitted" for item in entries[:3])
            and entries[3].status == "privately_contradicted"
        )
    )
    package_ids = [item.package.concept_id for item in library.entries]
    candidate_prediction_use_rate = float(np.mean([
        item.prediction_arm == "evidence_compiled_candidate" for item in results
    ]))
    gates = {
        "quality_partition_complete": (
            quality_count == private_spec.expected_quality_case_count
        ),
        "source_claim_compiler_chain_complete": all(
            item.compiled.evidence_hash == evidence.evidence_hash
            and item.compiled.package_hash == item.package.package_hash
            and item.compiled.static_checks_passed
            and item.compiled.numeric_checks_passed
            for item in library.entries
        ),
        "same_candidate_library_every_case": all(
            item.library_hash == library.library_hash
            and item.same_candidate_library_used
            for item in bundle.case_receipts
        ),
        "equal_expression_evaluation_budget": equal_budget,
        "all_attempts_persisted": all_persisted,
        "public_execution_private_blind": private_blind,
        "candidate_coverage": (
            sum(item.candidate_concept is not None for item in results) / len(results)
            >= private_spec.minimum_concept_recovery_accuracy
        ),
        "selected_concept_positive_influence": (
            positive_influence_rate >= private_spec.minimum_concept_recovery_accuracy
        ),
        "public_prediction_switch_complete": all(
            (
                item.prediction_arm == "evidence_compiled_candidate"
                and item.candidate_decision.selected_public_score
                <= item.baseline_decision.selected_public_score
            )
            or (
                item.prediction_arm == "polynomial_baseline"
                and (
                    item.candidate_decision.selected_public_score is None
                    or item.candidate_decision.selected_public_score
                    > item.baseline_decision.selected_public_score
                )
            )
            for item in bundle.case_receipts if not item.quality_flags
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
            for mechanism in MECHANISMS_V313
        ),
        "scaled_representation_non_regression": (
            representation_candidate["anonymous_scaled"]
            <= representation_candidate["anonymous_reference"]
            + private_spec.maximum_scaled_representation_regression
        ),
        "material_negative_transfer_controlled": (
            negative_upper <= private_spec.maximum_negative_transfer_upper_95
        ),
        "no_decoy_concept_selected": (
            selection_counts.get("affine_rate_decoy", 0) == 0
        ),
        "private_adjudication_statuses_correct": expected_statuses,
        "experience_active_view_correct": (
            experience.active_concept_versions == expected_active
        ),
        "candidate_library_complete": (
            set(selection_counts).issubset(set(package_ids))
            and len(library.entries) == 4
        ),
        "no_task_router_or_real_world_execution": True,
    }
    ready = private_spec.phase == "confirmation" and all(gates.values())
    status = (
        "evidence_concept_development_diagnostic_v313"
        if private_spec.phase == "development"
        else (
            "evidence_compiled_concepts_admitted_v313"
            if ready else "evidence_compiled_concepts_refuted_v313"
        )
    )
    report = EvidenceCompiledGrowthReportV313.seal(
        report_id=f"report_{private_spec.experiment_id}",
        phase=private_spec.phase,
        private_spec_hash=private_spec.spec_hash,
        public_protocol_hash=public_protocol.protocol_hash,
        public_pack_hash=private_pack.public_pack_hash,
        private_pack_hash=private_pack.pack_hash,
        bundle_hash=bundle.bundle_hash,
        library_hash=library.library_hash,
        lineage_receipt_hash=lineage.receipt_hash,
        source_v312_report_hash=lineage.source_report_hash,
        adjudication_hash=adjudication.adjudication_hash,
        input_experience_store_hash=input_store_hash,
        output_experience_store_hash=experience.store_hash,
        case_results=results,
        pair_results=pairs,
        performance_case_count=len(results),
        quality_case_count=quality_count,
        baseline_expression_evaluation_count=baseline_evaluations,
        candidate_expression_evaluation_count=candidate_evaluations,
        baseline_coverage=sum(item.baseline_concept is not None for item in results) / len(results),
        candidate_coverage=sum(item.candidate_concept is not None for item in results) / len(results),
        baseline_mean_target_loss=float(np.mean(baseline_losses)),
        concept_model_mean_target_loss=float(np.mean(concept_model_losses)),
        candidate_mean_target_loss=float(np.mean(candidate_losses)),
        paired_mean_improvement=float(np.mean(improvements)),
        paired_improvement_ci_lower=ci_lower,
        paired_improvement_ci_upper=ci_upper,
        concept_recovery_accuracy=recovery_accuracy,
        pair_concept_consistency=pair_consistency,
        maximum_pair_loss_difference=max_pair_difference,
        selected_positive_influence_rate=positive_influence_rate,
        candidate_prediction_use_rate=candidate_prediction_use_rate,
        mean_loss_by_mechanism_baseline=mechanism_baseline,
        mean_loss_by_mechanism_candidate=mechanism_candidate,
        mean_loss_by_representation_candidate=representation_candidate,
        candidate_selection_counts={
            concept_id: selection_counts.get(concept_id, 0)
            for concept_id in package_ids
        },
        material_negative_transfer_count=negatives,
        material_negative_transfer_upper_95=negative_upper,
        public_execution_private_blind=private_blind,
        all_attempts_persisted=all_persisted,
        equal_evaluation_budget=equal_budget,
        gates=gates,
        ready_for_concept_admission=ready,
        status=status,
        created_at=evaluated_at,
    )
    return GrowthEvaluationV313(adjudication, experience, report)


class EvidenceCompiledGrowthManifestV313(StrictModel):
    schema_version: Literal["3.13"] = "3.13"
    manifest_id: Identifier
    run_id: Identifier
    phase: Literal["development", "confirmation"]
    artifact_refs: list[ArtifactRef] = Field(min_length=12, max_length=12)
    terminal_status: Literal[
        "evidence_concept_development_diagnostic_v313",
        "evidence_compiled_concepts_admitted_v313",
        "evidence_compiled_concepts_refuted_v313",
    ]
    created_at: datetime
    manifest_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_manifest(self) -> "EvidenceCompiledGrowthManifestV313":
        _assert_timezone(self.created_at, "created_at")
        if self.manifest_hash and self.manifest_hash != self.content_hash():
            raise ValueError("V3.13 manifest hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "manifest_hash")

    def assert_sealed(self) -> None:
        if not self.manifest_hash or self.manifest_hash != self.content_hash():
            raise ValueError("V3.13 manifest is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "EvidenceCompiledGrowthManifestV313":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"manifest_hash"}),
            manifest_hash=draft.content_hash(),
        )


@dataclass(frozen=True)
class EvidenceCompiledGrowthOutcomeV313:
    store: RunStore
    public_pack: PublicGrowthWorldPackV313
    private_pack: PrivateGrowthWorldPackV313
    bundle: GrowthDiscoveryBundleV313
    adjudication: PrivateConceptAdjudicationV313
    experience_store: ConceptExperienceStoreV313
    report: EvidenceCompiledGrowthReportV313
    manifest: EvidenceCompiledGrowthManifestV313


def _load_development_v313(
    run_directory: str | Path,
    *,
    source_v312_run_directory: str | Path,
) -> tuple[EvidenceCompiledGrowthReportV313, ConceptExperienceStoreV313]:
    if not verify_evidence_compiled_growth_run_v313(
        run_directory,
        source_v312_run_directory=source_v312_run_directory,
        development_run_directory=None,
    ):
        raise ValueError("V3.13 development run did not independently verify")
    store = RunStore.open_existing(run_directory)
    refs = _committed_refs(store)
    report = _load_one(
        store, refs, "evidence_compiled_growth_report_v313",
        EvidenceCompiledGrowthReportV313,
    )
    experience = _load_one(
        store, refs, "concept_experience_store_v313", ConceptExperienceStoreV313
    )
    if report.phase != "development":
        raise ValueError("V3.13 development lineage phase differs")
    return report, experience


def run_evidence_compiled_growth_worldpack_v313(
    output_root: str | Path,
    *,
    source_v312_run_directory: str | Path,
    development_run_directory: str | Path | None = None,
    evidence: ConceptEvidenceBundleV313,
    lineage: VerifiedV312LineageReceiptV313,
    library: CompiledConceptLibraryV313,
    policy: EvidenceConceptPolicyV313,
    public_protocol: PublicGrowthProtocolV313,
    private_spec: PrivateGrowthWorldPackSpecV313,
    evaluated_at: datetime | None = None,
    run_id: str | None = None,
) -> EvidenceCompiledGrowthOutcomeV313:
    if not verify_v312_lineage_receipt_v313(
        lineage, source_v312_run_directory
    ):
        raise ValueError("V3.13 source lineage receipt failed quick verification")
    development_report = None
    development_experience = None
    if private_spec.phase == "confirmation":
        if development_run_directory is None:
            raise ValueError("V3.13 confirmation requires development run")
        development_report, development_experience = _load_development_v313(
            development_run_directory,
            source_v312_run_directory=source_v312_run_directory,
        )
    for item in (
        evidence, lineage, library, policy, public_protocol, private_spec,
    ):
        item.assert_sealed()
    if (
        evidence.evidence_hash != library.evidence_hash
        or evidence.evidence_hash != policy.evidence_hash
        or evidence.evidence_hash != public_protocol.evidence_hash
        or lineage.receipt_hash != policy.lineage_receipt_hash
        or lineage.receipt_hash != public_protocol.lineage_receipt_hash
        or lineage.receipt_hash != private_spec.lineage_receipt_hash
        or library.library_hash != policy.library_hash
        or library.library_hash != public_protocol.library_hash
        or library.library_hash != private_spec.library_hash
        or policy.policy_hash != public_protocol.policy_hash
        or public_protocol.protocol_hash != private_spec.public_protocol_hash
        or private_spec.frozen_at < public_protocol.frozen_at
        or (
            private_spec.phase == "confirmation"
            and (
                development_report.report_hash
                != private_spec.development_report_hash
                or development_experience.store_hash
                != private_spec.development_experience_store_hash
                or private_spec.frozen_at < development_report.created_at
                or private_spec.frozen_at < development_experience.updated_at
            )
        )
    ):
        raise ValueError("V3.13 frozen lineage binding differs")
    at = evaluated_at or datetime.now(timezone.utc)
    wall_now = datetime.now(timezone.utc)
    if at < private_spec.frozen_at:
        raise ValueError("V3.13 evaluation predates private spec")
    if (
        private_spec.frozen_at > wall_now + timedelta(minutes=5)
        or at > wall_now + timedelta(minutes=5)
    ):
        raise ValueError("V3.13 audit timestamp is implausibly in the future")
    store = RunStore(
        output_root,
        run_id=run_id or f"evidence-growth-v313-{uuid4().hex[:10]}",
    )
    refs = [
        store.put_artifact("concept_evidence_bundle_v313", evidence),
        store.put_artifact("verified_v312_lineage_receipt_v313", lineage),
        store.put_artifact("compiled_concept_library_v313", library),
        store.put_artifact("evidence_concept_policy_v313", policy),
        store.put_artifact("public_growth_protocol_v313", public_protocol),
        store.put_artifact("private_growth_worldpack_spec_v313", private_spec),
    ]
    store.emit("evidence_compiled_growth_v313_protocol_frozen", {
        "phase": private_spec.phase,
        "private_spec_hash": private_spec.spec_hash,
        "development_report_hash": private_spec.development_report_hash,
        "development_experience_store_hash": (
            private_spec.development_experience_store_hash
        ),
        "private_pack_not_passed_to_generator": True,
        "source_code_execution_permitted": False,
        "public_score_can_admit_concept": False,
    })
    public_pack, private_pack = generate_growth_worldpacks_v313(
        private_spec, public_protocol, library, generated_at=at
    )
    bundle = execute_evidence_compiled_growth_v313(
        public_protocol, public_pack, policy, library, executed_at=at
    )
    evaluation = evaluate_evidence_compiled_growth_v313(
        private_spec,
        public_protocol,
        lineage,
        evidence,
        library,
        private_pack,
        bundle,
        evaluated_at=at,
        input_experience_store=development_experience,
    )
    refs.extend([
        store.put_artifact("public_growth_worldpack_v313", public_pack),
        store.put_artifact("private_growth_worldpack_v313", private_pack),
        store.put_artifact("growth_discovery_bundle_v313", bundle),
        store.put_artifact(
            "private_concept_adjudication_v313", evaluation.adjudication
        ),
        store.put_artifact(
            "concept_experience_store_v313", evaluation.experience_store
        ),
        store.put_artifact(
            "evidence_compiled_growth_report_v313", evaluation.report
        ),
    ])
    manifest = EvidenceCompiledGrowthManifestV313.seal(
        manifest_id=f"manifest_{store.run_id}",
        run_id=store.run_id,
        phase=private_spec.phase,
        artifact_refs=refs,
        terminal_status=evaluation.report.status,
        created_at=at,
    )
    manifest_ref = store.put_artifact(
        "evidence_compiled_growth_manifest_v313", manifest
    )
    store.emit("evidence_compiled_growth_v313_adjudicated", {
        "manifest_ref": manifest_ref.model_dump(mode="json")
    })
    if not verify_evidence_compiled_growth_run_v313(
        store.run_directory,
        source_v312_run_directory=source_v312_run_directory,
        development_run_directory=development_run_directory,
    ):
        raise RuntimeError("V3.13 run failed independent verification")
    return EvidenceCompiledGrowthOutcomeV313(
        store=store,
        public_pack=public_pack,
        private_pack=private_pack,
        bundle=bundle,
        adjudication=evaluation.adjudication,
        experience_store=evaluation.experience_store,
        report=evaluation.report,
        manifest=manifest,
    )


def verify_evidence_compiled_growth_run_v313(
    run_directory: str | Path,
    *,
    source_v312_run_directory: str | Path,
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
        if len(refs) != 13:
            return False
        for ref in refs:
            store.load_artifact(ref)
        evidence = _load_one(
            store, refs, "concept_evidence_bundle_v313", ConceptEvidenceBundleV313
        )
        lineage = _load_one(
            store, refs, "verified_v312_lineage_receipt_v313",
            VerifiedV312LineageReceiptV313,
        )
        library = _load_one(
            store, refs, "compiled_concept_library_v313",
            CompiledConceptLibraryV313,
        )
        policy = _load_one(
            store, refs, "evidence_concept_policy_v313", EvidenceConceptPolicyV313
        )
        protocol = _load_one(
            store, refs, "public_growth_protocol_v313", PublicGrowthProtocolV313
        )
        spec = _load_one(
            store, refs, "private_growth_worldpack_spec_v313",
            PrivateGrowthWorldPackSpecV313,
        )
        public_pack = _load_one(
            store, refs, "public_growth_worldpack_v313", PublicGrowthWorldPackV313
        )
        private_pack = _load_one(
            store, refs, "private_growth_worldpack_v313", PrivateGrowthWorldPackV313
        )
        bundle = _load_one(
            store, refs, "growth_discovery_bundle_v313", GrowthDiscoveryBundleV313
        )
        adjudication = _load_one(
            store, refs, "private_concept_adjudication_v313",
            PrivateConceptAdjudicationV313,
        )
        experience = _load_one(
            store, refs, "concept_experience_store_v313", ConceptExperienceStoreV313
        )
        report = _load_one(
            store, refs, "evidence_compiled_growth_report_v313",
            EvidenceCompiledGrowthReportV313,
        )
        manifest = _load_one(
            store, refs, "evidence_compiled_growth_manifest_v313",
            EvidenceCompiledGrowthManifestV313,
        )
        for item in (
            evidence, lineage, library, policy, protocol, spec, public_pack,
            private_pack, bundle, adjudication, experience, report, manifest,
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
        development_experience = None
        if spec.phase == "confirmation":
            if development_run_directory is None:
                return False
            if Path(development_run_directory).resolve() == Path(run_directory).resolve():
                return False
            development_report, development_experience = _load_development_v313(
                development_run_directory,
                source_v312_run_directory=source_v312_run_directory,
            )
        if (
            not verify_v312_lineage_receipt_v313(
                lineage, source_v312_run_directory
            )
            or evidence.evidence_hash != library.evidence_hash
            or evidence.evidence_hash != policy.evidence_hash
            or evidence.evidence_hash != protocol.evidence_hash
            or lineage.receipt_hash != policy.lineage_receipt_hash
            or lineage.receipt_hash != protocol.lineage_receipt_hash
            or lineage.receipt_hash != spec.lineage_receipt_hash
            or library.library_hash != policy.library_hash
            or library.library_hash != protocol.library_hash
            or library.library_hash != spec.library_hash
            or policy.policy_hash != protocol.policy_hash
            or protocol.protocol_hash != spec.public_protocol_hash
            or (
                spec.phase == "confirmation"
                and (
                    development_report.report_hash != spec.development_report_hash
                    or development_experience.store_hash
                    != spec.development_experience_store_hash
                    or spec.frozen_at < development_report.created_at
                    or spec.frozen_at < development_experience.updated_at
                )
            )
        ):
            return False
        rebuilt_evidence = default_concept_evidence_v313(
            v312_report_hash=lineage.source_report_hash
        )
        rebuilt_library = compile_default_concept_library_v313(rebuilt_evidence)
        if (
            rebuilt_evidence.evidence_hash != evidence.evidence_hash
            or rebuilt_library.library_hash != library.library_hash
        ):
            return False
        regenerated_public, regenerated_private = generate_growth_worldpacks_v313(
            spec, protocol, library, generated_at=private_pack.generated_at
        )
        if (
            regenerated_public.public_pack_hash != public_pack.public_pack_hash
            or regenerated_private.pack_hash != private_pack.pack_hash
        ):
            return False
        recomputed_bundle = execute_evidence_compiled_growth_v313(
            protocol, public_pack, policy, library, executed_at=bundle.created_at
        )
        if recomputed_bundle.bundle_hash != bundle.bundle_hash:
            return False
        recomputed = evaluate_evidence_compiled_growth_v313(
            spec,
            protocol,
            lineage,
            evidence,
            library,
            private_pack,
            bundle,
            evaluated_at=report.created_at,
            input_experience_store=development_experience,
        )
        if (
            recomputed.adjudication.adjudication_hash
            != adjudication.adjudication_hash
            or recomputed.experience_store.store_hash != experience.store_hash
            or recomputed.report.report_hash != report.report_hash
        ):
            return False
        if (
            manifest.phase != spec.phase
            or manifest.terminal_status != report.status
            or [item.model_dump(mode="json") for item in manifest.artifact_refs]
            != [item.model_dump(mode="json") for item in refs[:12]]
        ):
            return False
        event_types = [event["event_type"] for event in events]
        freeze_index = event_types.index(
            "evidence_compiled_growth_v313_protocol_frozen"
        )
        private_index = next(
            index for index, event in enumerate(events)
            if event["event_type"] == "artifact_committed"
            and event["payload"]["kind"] == "private_growth_worldpack_v313"
        )
        return freeze_index < private_index
    except (
        OSError, RuntimeError, ValueError, KeyError, TypeError,
        json.JSONDecodeError, FloatingPointError, AttributeError,
        np.linalg.LinAlgError,
    ):
        return False
