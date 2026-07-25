from __future__ import annotations

import hashlib
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from fma.schemas import StrictModel
from fma.v2.schemas import Identifier, Sha256, _assert_timezone

from .model_challenge_v37 import _hash_without


SourceTierV313 = Literal[
    "primary_paper",
    "normative_standard",
    "official_software",
    "prior_private_admission",
]
ClaimKindV313 = Literal[
    "operator_template",
    "parameter_role",
    "unit_relation",
    "scope_condition",
    "known_limitation",
    "compiler_constraint",
]
NodeKindV313 = Literal[
    "state",
    "parameter",
    "constant",
    "add",
    "subtract",
    "multiply",
    "divide",
    "log",
    "power",
]
ExperienceEventTypeV313 = Literal[
    "proposed",
    "compiled",
    "rejected_static",
    "rejected_numeric",
    "development_supported",
    "privately_admitted",
    "contradicted",
    "revoked",
]


def _file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class DimensionV313(StrictModel):
    state_power: Annotated[int, Field(ge=-8, le=8)] = 0
    time_power: Annotated[int, Field(ge=-8, le=8)] = 0

    def multiply(self, other: "DimensionV313") -> "DimensionV313":
        return DimensionV313(
            state_power=self.state_power + other.state_power,
            time_power=self.time_power + other.time_power,
        )

    def divide(self, other: "DimensionV313") -> "DimensionV313":
        return DimensionV313(
            state_power=self.state_power - other.state_power,
            time_power=self.time_power - other.time_power,
        )

    def power(self, exponent: int) -> "DimensionV313":
        return DimensionV313(
            state_power=self.state_power * exponent,
            time_power=self.time_power * exponent,
        )

    @property
    def is_dimensionless(self) -> bool:
        return self.state_power == 0 and self.time_power == 0


DIMENSIONLESS_V313 = DimensionV313()
STATE_DIMENSION_V313 = DimensionV313(state_power=1)
TIME_DIMENSION_V313 = DimensionV313(time_power=1)
RHS_DIMENSION_V313 = DimensionV313(state_power=1, time_power=-1)


class SourceRecordV313(StrictModel):
    schema_version: Literal["3.13"] = "3.13"
    source_id: Identifier
    title: str = Field(min_length=5)
    persistent_id: str = Field(min_length=3)
    canonical_url: str = Field(min_length=10)
    source_tier: SourceTierV313
    retrieved_on: Literal["2026-07-22"] = "2026-07-22"
    support_summary: str = Field(min_length=10)
    limitation_summary: str = Field(min_length=10)
    full_content_snapshot_available: Literal[False] = False
    execution_permission: Literal[False] = False
    record_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_record(self) -> "SourceRecordV313":
        if self.record_hash and self.record_hash != self.content_hash():
            raise ValueError("V3.13 source record hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "record_hash")

    def assert_sealed(self) -> None:
        if not self.record_hash or self.record_hash != self.content_hash():
            raise ValueError("V3.13 source record is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "SourceRecordV313":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"record_hash"}),
            record_hash=draft.content_hash(),
        )


class ScientificClaimV313(StrictModel):
    schema_version: Literal["3.13"] = "3.13"
    claim_id: Identifier
    source_record_hash: Sha256
    claim_kind: ClaimKindV313
    locator: str = Field(min_length=3)
    claim_summary: str = Field(min_length=10)
    extraction_method: Literal["sol_research_draft"] = "sol_research_draft"
    authoritative_extraction: Literal[False] = False
    execution_permission: Literal[False] = False
    contradiction_claim_hashes: list[Sha256] = Field(default_factory=list)
    claim_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_claim(self) -> "ScientificClaimV313":
        if self.claim_hash and self.claim_hash != self.content_hash():
            raise ValueError("V3.13 scientific claim hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "claim_hash")

    def assert_sealed(self) -> None:
        if not self.claim_hash or self.claim_hash != self.content_hash():
            raise ValueError("V3.13 scientific claim is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ScientificClaimV313":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"claim_hash"}),
            claim_hash=draft.content_hash(),
        )


class ConceptEvidenceBundleV313(StrictModel):
    schema_version: Literal["3.13"] = "3.13"
    evidence_id: Identifier
    sources: list[SourceRecordV313] = Field(min_length=6)
    claims: list[ScientificClaimV313] = Field(min_length=6)
    retrieval_scope: Literal[
        "targeted_primary_normative_official_not_systematic_review"
    ] = "targeted_primary_normative_official_not_systematic_review"
    evidence_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_evidence(self) -> "ConceptEvidenceBundleV313":
        source_hashes = set()
        source_ids = set()
        for source in self.sources:
            source.assert_sealed()
            if source.source_id in source_ids or source.record_hash in source_hashes:
                raise ValueError("V3.13 source records differ")
            source_ids.add(source.source_id)
            source_hashes.add(source.record_hash)
        claim_hashes = set()
        claim_ids = set()
        for claim in self.claims:
            claim.assert_sealed()
            if claim.source_record_hash not in source_hashes:
                raise ValueError("V3.13 claim source is unavailable")
            if claim.claim_id in claim_ids or claim.claim_hash in claim_hashes:
                raise ValueError("V3.13 scientific claims differ")
            claim_ids.add(claim.claim_id)
            claim_hashes.add(claim.claim_hash)
        if any(
            contradiction not in claim_hashes
            for claim in self.claims
            for contradiction in claim.contradiction_claim_hashes
        ):
            raise ValueError("V3.13 contradiction claim is unavailable")
        if self.evidence_hash and self.evidence_hash != self.content_hash():
            raise ValueError("V3.13 evidence bundle hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "evidence_hash")

    def assert_sealed(self) -> None:
        if not self.evidence_hash or self.evidence_hash != self.content_hash():
            raise ValueError("V3.13 evidence bundle is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ConceptEvidenceBundleV313":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"evidence_hash"}),
            evidence_hash=draft.content_hash(),
        )


class ParameterSpecV313(StrictModel):
    parameter_id: Identifier
    dimension: DimensionV313
    lower_bound: Annotated[float, Field(allow_inf_nan=False)]
    upper_bound: Annotated[float, Field(allow_inf_nan=False)]
    initial_value: Annotated[float, Field(allow_inf_nan=False)]

    @model_validator(mode="after")
    def validate_parameter(self) -> "ParameterSpecV313":
        if not self.lower_bound < self.upper_bound:
            raise ValueError("V3.13 parameter bounds must increase")
        if not self.lower_bound <= self.initial_value <= self.upper_bound:
            raise ValueError("V3.13 parameter initial value is out of bounds")
        return self


class OperatorNodeV313(StrictModel):
    kind: NodeKindV313
    state_index: Annotated[int, Field(ge=0, le=0)] | None = None
    parameter_id: Identifier | None = None
    constant_value: Annotated[float, Field(allow_inf_nan=False)] | None = None
    children: list["OperatorNodeV313"] = Field(default_factory=list, max_length=2)

    @model_validator(mode="after")
    def validate_node(self) -> "OperatorNodeV313":
        arity = {
            "state": 0,
            "parameter": 0,
            "constant": 0,
            "add": 2,
            "subtract": 2,
            "multiply": 2,
            "divide": 2,
            "log": 1,
            "power": 2,
        }[self.kind]
        if len(self.children) != arity:
            raise ValueError("V3.13 operator arity differs")
        if (self.state_index is not None) != (self.kind == "state"):
            raise ValueError("V3.13 state node payload differs")
        if (self.parameter_id is not None) != (self.kind == "parameter"):
            raise ValueError("V3.13 parameter node payload differs")
        if (self.constant_value is not None) != (self.kind == "constant"):
            raise ValueError("V3.13 constant node payload differs")
        return self


class ConceptPackageV313(StrictModel):
    schema_version: Literal["3.13"] = "3.13"
    concept_id: Identifier
    concept_version: Literal[1] = 1
    evidence_hash: Sha256
    supporting_claim_hashes: list[Sha256] = Field(min_length=1)
    state_dimension: Literal[1] = 1
    state_unit: DimensionV313 = STATE_DIMENSION_V313
    time_unit: DimensionV313 = TIME_DIMENSION_V313
    rhs: OperatorNodeV313
    parameters: list[ParameterSpecV313] = Field(min_length=1, max_length=6)
    state_domain_lower: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    state_domain_upper: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    maximum_ast_nodes: Literal[24] = 24
    maximum_ast_depth: Literal[7] = 7
    arbitrary_code_present: Literal[False] = False
    custom_operator_present: Literal[False] = False
    package_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_package(self) -> "ConceptPackageV313":
        if not self.state_domain_lower < self.state_domain_upper:
            raise ValueError("V3.13 state domain must increase")
        parameter_ids = [item.parameter_id for item in self.parameters]
        if len(parameter_ids) != len(set(parameter_ids)):
            raise ValueError("V3.13 parameter ids differ")
        referenced = {
            node.parameter_id
            for node in _walk_nodes_v313(self.rhs)
            if node.kind == "parameter"
        }
        if referenced != set(parameter_ids):
            raise ValueError("V3.13 parameter declarations and AST differ")
        if len(self.supporting_claim_hashes) != len(set(self.supporting_claim_hashes)):
            raise ValueError("V3.13 supporting claims differ")
        if self.package_hash and self.package_hash != self.content_hash():
            raise ValueError("V3.13 concept package hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "package_hash")

    def assert_sealed(self) -> None:
        if not self.package_hash or self.package_hash != self.content_hash():
            raise ValueError("V3.13 concept package is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ConceptPackageV313":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"package_hash"}),
            package_hash=draft.content_hash(),
        )


def _walk_nodes_v313(root: OperatorNodeV313) -> list[OperatorNodeV313]:
    nodes = [root]
    for child in root.children:
        nodes.extend(_walk_nodes_v313(child))
    return nodes


def _node_depth_v313(root: OperatorNodeV313) -> int:
    if not root.children:
        return 1
    return 1 + max(_node_depth_v313(child) for child in root.children)


class UnitProofStepV313(StrictModel):
    path: str = Field(min_length=1)
    node_kind: NodeKindV313
    inferred_dimension: DimensionV313


class NumericCanaryReceiptV313(StrictModel):
    evaluated_point_count: Annotated[int, Field(ge=1)]
    finite_point_count: Annotated[int, Field(ge=0)]
    maximum_absolute_value: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    domain_violation_count: Annotated[int, Field(ge=0)]
    passed: bool


class CompiledConceptV313(StrictModel):
    schema_version: Literal["3.13"] = "3.13"
    compiled_id: Identifier
    concept_id: Identifier
    concept_version: Literal[1] = 1
    evidence_hash: Sha256
    package_hash: Sha256
    compiler_source_hash: Sha256
    ast_node_count: Annotated[int, Field(ge=1, le=24)]
    ast_depth: Annotated[int, Field(ge=1, le=7)]
    rhs_dimension: DimensionV313
    unit_proof: list[UnitProofStepV313] = Field(min_length=1, max_length=24)
    numeric_canary: NumericCanaryReceiptV313
    arbitrary_code_executed: Literal[False] = False
    custom_operator_executed: Literal[False] = False
    static_checks_passed: Literal[True] = True
    numeric_checks_passed: Literal[True] = True
    compiled_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_compiled(self) -> "CompiledConceptV313":
        if self.rhs_dimension != RHS_DIMENSION_V313:
            raise ValueError("V3.13 compiled RHS dimension differs")
        if not self.numeric_canary.passed:
            raise ValueError("V3.13 compiled concept needs numeric canaries")
        if self.compiled_hash and self.compiled_hash != self.content_hash():
            raise ValueError("V3.13 compiled concept hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "compiled_hash")

    def assert_sealed(self) -> None:
        if not self.compiled_hash or self.compiled_hash != self.content_hash():
            raise ValueError("V3.13 compiled concept is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "CompiledConceptV313":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"compiled_hash"}),
            compiled_hash=draft.content_hash(),
        )


def _infer_dimension_v313(
    node: OperatorNodeV313,
    parameters: dict[str, ParameterSpecV313],
    *,
    path: str,
    proof: list[UnitProofStepV313],
) -> DimensionV313:
    child_dimensions = [
        _infer_dimension_v313(
            child,
            parameters,
            path=f"{path}.{index}",
            proof=proof,
        )
        for index, child in enumerate(node.children)
    ]
    if node.kind == "state":
        result = STATE_DIMENSION_V313
    elif node.kind == "parameter":
        if node.parameter_id not in parameters:
            raise ValueError("V3.13 AST references an unknown parameter")
        result = parameters[node.parameter_id].dimension
    elif node.kind == "constant":
        result = DIMENSIONLESS_V313
    elif node.kind in ("add", "subtract"):
        if child_dimensions[0] != child_dimensions[1]:
            raise ValueError("V3.13 add/subtract requires equal dimensions")
        result = child_dimensions[0]
    elif node.kind == "multiply":
        result = child_dimensions[0].multiply(child_dimensions[1])
    elif node.kind == "divide":
        result = child_dimensions[0].divide(child_dimensions[1])
    elif node.kind == "log":
        if not child_dimensions[0].is_dimensionless:
            raise ValueError("V3.13 log requires a dimensionless argument")
        result = DIMENSIONLESS_V313
    elif node.kind == "power":
        if not child_dimensions[1].is_dimensionless:
            raise ValueError("V3.13 power exponent must be dimensionless")
        # A variable exponent is safe only for a dimensionless base.  Integer
        # constant exponents may scale a dimensional base.
        exponent_node = node.children[1]
        if exponent_node.kind == "constant":
            exponent = exponent_node.constant_value
            if not float(exponent).is_integer():
                if not child_dimensions[0].is_dimensionless:
                    raise ValueError(
                        "V3.13 fractional power requires a dimensionless base"
                    )
                result = DIMENSIONLESS_V313
            else:
                result = child_dimensions[0].power(int(exponent))
        else:
            if not child_dimensions[0].is_dimensionless:
                raise ValueError(
                    "V3.13 variable power requires a dimensionless base"
                )
            result = DIMENSIONLESS_V313
    else:  # pragma: no cover - Literal and Pydantic make this unreachable.
        raise ValueError("V3.13 unsupported operator")
    proof.append(UnitProofStepV313(
        path=path,
        node_kind=node.kind,
        inferred_dimension=result,
    ))
    return result


def evaluate_operator_ast_v313(
    node: OperatorNodeV313,
    state: list[float],
    parameters: dict[str, float],
) -> float:
    if node.kind == "state":
        return float(state[node.state_index])
    if node.kind == "parameter":
        return float(parameters[node.parameter_id])
    if node.kind == "constant":
        return float(node.constant_value)
    values = [
        evaluate_operator_ast_v313(child, state, parameters)
        for child in node.children
    ]
    if node.kind == "add":
        return values[0] + values[1]
    if node.kind == "subtract":
        return values[0] - values[1]
    if node.kind == "multiply":
        return values[0] * values[1]
    if node.kind == "divide":
        if abs(values[1]) < 1e-12:
            raise ValueError("V3.13 division canary reached zero")
        return values[0] / values[1]
    if node.kind == "log":
        if values[0] <= 0:
            raise ValueError("V3.13 log canary left its positive domain")
        return math.log(values[0])
    if node.kind == "power":
        if values[0] <= 0 and not float(values[1]).is_integer():
            raise ValueError("V3.13 fractional power left its real domain")
        return values[0] ** values[1]
    raise ValueError("V3.13 unsupported operator")


def compile_concept_package_v313(
    package: ConceptPackageV313,
    evidence: ConceptEvidenceBundleV313,
) -> CompiledConceptV313:
    package.assert_sealed()
    evidence.assert_sealed()
    claim_hashes = {item.claim_hash for item in evidence.claims}
    if package.evidence_hash != evidence.evidence_hash:
        raise ValueError("V3.13 package evidence binding differs")
    if any(item not in claim_hashes for item in package.supporting_claim_hashes):
        raise ValueError("V3.13 package supporting claim is unavailable")
    nodes = _walk_nodes_v313(package.rhs)
    depth = _node_depth_v313(package.rhs)
    if len(nodes) > package.maximum_ast_nodes:
        raise ValueError("V3.13 AST node budget exceeded")
    if depth > package.maximum_ast_depth:
        raise ValueError("V3.13 AST depth budget exceeded")
    parameter_specs = {item.parameter_id: item for item in package.parameters}
    proof: list[UnitProofStepV313] = []
    rhs_dimension = _infer_dimension_v313(
        package.rhs,
        parameter_specs,
        path="rhs",
        proof=proof,
    )
    if rhs_dimension != RHS_DIMENSION_V313:
        raise ValueError("V3.13 RHS must have state/time units")
    state_points = [
        package.state_domain_lower,
        math.sqrt(package.state_domain_lower * package.state_domain_upper),
        package.state_domain_upper,
    ]
    parameter_points = []
    for selector in ("lower_bound", "initial_value", "upper_bound"):
        parameter_points.append({
            item.parameter_id: float(getattr(item, selector))
            for item in package.parameters
        })
    finite = 0
    violations = 0
    maximum = 0.0
    for state_value in state_points:
        for parameter_values in parameter_points:
            try:
                value = evaluate_operator_ast_v313(
                    package.rhs, [state_value], parameter_values
                )
                if not math.isfinite(value) or abs(value) > 1e6:
                    violations += 1
                else:
                    finite += 1
                    maximum = max(maximum, abs(value))
            except (ValueError, OverflowError, ZeroDivisionError):
                violations += 1
    canary = NumericCanaryReceiptV313(
        evaluated_point_count=len(state_points) * len(parameter_points),
        finite_point_count=finite,
        maximum_absolute_value=maximum,
        domain_violation_count=violations,
        passed=(violations == 0),
    )
    if not canary.passed:
        raise ValueError("V3.13 numeric domain canary failed")
    return CompiledConceptV313.seal(
        compiled_id=f"compiled_{package.concept_id}_v{package.concept_version}",
        concept_id=package.concept_id,
        concept_version=package.concept_version,
        evidence_hash=evidence.evidence_hash,
        package_hash=package.package_hash,
        compiler_source_hash=_file_sha256(__file__),
        ast_node_count=len(nodes),
        ast_depth=depth,
        rhs_dimension=rhs_dimension,
        unit_proof=proof,
        numeric_canary=canary,
    )


class ConceptExperienceEventV313(StrictModel):
    schema_version: Literal["3.13"] = "3.13"
    event_id: Identifier
    sequence: Annotated[int, Field(ge=1)]
    previous_event_hash: Sha256 | None
    event_type: ExperienceEventTypeV313
    concept_id: Identifier
    concept_version: Literal[1] = 1
    package_hash: Sha256
    compiled_hash: Sha256 | None = None
    attempt_hash: Sha256 | None = None
    adjudication_hash: Sha256 | None = None
    phase: Literal["research", "development", "confirmation", "post_confirmation"]
    private_evaluator_event: bool
    public_score_used_for_admission: Literal[False] = False
    generator_receives_private_feedback: Literal[False] = False
    created_at: datetime
    event_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_event(self) -> "ConceptExperienceEventV313":
        _assert_timezone(self.created_at, "created_at")
        private_types = {"privately_admitted", "contradicted", "revoked"}
        if self.event_type in private_types:
            if not self.private_evaluator_event or not self.adjudication_hash:
                raise ValueError("V3.13 private experience event needs adjudication")
        elif self.private_evaluator_event:
            raise ValueError("V3.13 public experience event cannot be private")
        if self.event_type in {
            "compiled", "development_supported", "privately_admitted",
            "contradicted", "revoked",
        } and not self.compiled_hash:
            raise ValueError("V3.13 experience event needs compiled lineage")
        if self.event_hash and self.event_hash != self.content_hash():
            raise ValueError("V3.13 experience event hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "event_hash")

    def assert_sealed(self) -> None:
        if not self.event_hash or self.event_hash != self.content_hash():
            raise ValueError("V3.13 experience event is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ConceptExperienceEventV313":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"event_hash"}),
            event_hash=draft.content_hash(),
        )


class ConceptExperienceStoreV313(StrictModel):
    schema_version: Literal["3.13"] = "3.13"
    store_id: Identifier
    evidence_hash: Sha256
    events: list[ConceptExperienceEventV313]
    head_event_hash: Sha256 | None
    active_concept_versions: dict[Identifier, int]
    created_at: datetime
    updated_at: datetime
    store_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_store(self) -> "ConceptExperienceStoreV313":
        _assert_timezone(self.created_at, "created_at")
        _assert_timezone(self.updated_at, "updated_at")
        if self.updated_at < self.created_at:
            raise ValueError("V3.13 experience store time differs")
        previous = None
        latest: dict[tuple[str, int], ConceptExperienceEventV313] = {}
        for index, event in enumerate(self.events, start=1):
            event.assert_sealed()
            if event.sequence != index or event.previous_event_hash != previous:
                raise ValueError("V3.13 experience chain differs")
            previous = event.event_hash
            latest[(event.concept_id, event.concept_version)] = event
        if self.head_event_hash != previous:
            raise ValueError("V3.13 experience head differs")
        expected_active = {
            concept_id: version
            for (concept_id, version), event in latest.items()
            if event.event_type == "privately_admitted"
        }
        if self.active_concept_versions != expected_active:
            raise ValueError("V3.13 active concept view differs")
        if self.store_hash and self.store_hash != self.content_hash():
            raise ValueError("V3.13 experience store hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "store_hash")

    def assert_sealed(self) -> None:
        if not self.store_hash or self.store_hash != self.content_hash():
            raise ValueError("V3.13 experience store is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ConceptExperienceStoreV313":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"store_hash"}),
            store_hash=draft.content_hash(),
        )


def empty_concept_experience_store_v313(
    evidence: ConceptEvidenceBundleV313,
    *,
    created_at: datetime | None = None,
) -> ConceptExperienceStoreV313:
    evidence.assert_sealed()
    at = created_at or datetime.now(timezone.utc)
    return ConceptExperienceStoreV313.seal(
        store_id="cross_task_concept_experience_v313",
        evidence_hash=evidence.evidence_hash,
        events=[],
        head_event_hash=None,
        active_concept_versions={},
        created_at=at,
        updated_at=at,
    )


def append_concept_experience_event_v313(
    store: ConceptExperienceStoreV313,
    *,
    event_type: ExperienceEventTypeV313,
    package: ConceptPackageV313,
    compiled: CompiledConceptV313 | None,
    phase: Literal["research", "development", "confirmation", "post_confirmation"],
    created_at: datetime,
    attempt_hash: str | None = None,
    adjudication_hash: str | None = None,
    private_evaluator_event: bool = False,
) -> ConceptExperienceStoreV313:
    store.assert_sealed()
    package.assert_sealed()
    if package.evidence_hash != store.evidence_hash:
        raise ValueError("V3.13 experience evidence binding differs")
    if compiled:
        compiled.assert_sealed()
        if compiled.package_hash != package.package_hash:
            raise ValueError("V3.13 compiled experience binding differs")
    if created_at < store.updated_at:
        raise ValueError("V3.13 experience event predates store")
    event = ConceptExperienceEventV313.seal(
        event_id=f"experience_{len(store.events) + 1}_{package.concept_id}",
        sequence=len(store.events) + 1,
        previous_event_hash=store.head_event_hash,
        event_type=event_type,
        concept_id=package.concept_id,
        concept_version=package.concept_version,
        package_hash=package.package_hash,
        compiled_hash=(compiled.compiled_hash if compiled else None),
        attempt_hash=attempt_hash,
        adjudication_hash=adjudication_hash,
        phase=phase,
        private_evaluator_event=private_evaluator_event,
        created_at=created_at,
    )
    events = [*store.events, event]
    latest: dict[tuple[str, int], ConceptExperienceEventV313] = {}
    for item in events:
        latest[(item.concept_id, item.concept_version)] = item
    active = {
        concept_id: version
        for (concept_id, version), item in latest.items()
        if item.event_type == "privately_admitted"
    }
    return ConceptExperienceStoreV313.seal(
        store_id=store.store_id,
        evidence_hash=store.evidence_hash,
        events=events,
        head_event_hash=event.event_hash,
        active_concept_versions=active,
        created_at=store.created_at,
        updated_at=created_at,
    )


def _state() -> OperatorNodeV313:
    return OperatorNodeV313(kind="state", state_index=0)


def _parameter(parameter_id: str) -> OperatorNodeV313:
    return OperatorNodeV313(kind="parameter", parameter_id=parameter_id)


def _constant(value: float) -> OperatorNodeV313:
    return OperatorNodeV313(kind="constant", constant_value=value)


def _operator(kind: NodeKindV313, *children: OperatorNodeV313) -> OperatorNodeV313:
    return OperatorNodeV313(kind=kind, children=list(children))


def default_concept_evidence_v313(
    *,
    v312_report_hash: str,
) -> ConceptEvidenceBundleV313:
    sources = [
        SourceRecordV313.seal(
            source_id="llm_sr_2025",
            title="LLM-SR Scientific Equation Discovery via Programming with Large Language Models",
            persistent_id="arXiv:2404.18400v3",
            canonical_url="https://arxiv.org/abs/2404.18400",
            source_tier="primary_paper",
            support_summary="Treat equation skeletons as mathematical programs and fit their parameters separately.",
            limitation_summary="Benchmark performance does not make generated programs safe or source-faithful.",
        ),
        SourceRecordV313.seal(
            source_id="lasr_2024",
            title="Symbolic Regression with a Learned Concept Library",
            persistent_id="doi:10.52202/079017-1419",
            canonical_url="https://proceedings.neurips.cc/paper_files/paper/2024/hash/4ec3ddc465c6d650c9c419fb91f1c00a-Abstract-Conference.html",
            source_tier="primary_paper",
            support_summary="Maintain abstract concepts separately from the ordinary hypothesis population.",
            limitation_summary="Textual concepts are neither executable nor independently validated models.",
        ),
        SourceRecordV313.seal(
            source_id="igsr_2026",
            title="Influence-Guided Symbolic Regression",
            persistent_id="arXiv:2605.29184v1",
            canonical_url="https://arxiv.org/abs/2605.29184",
            source_tier="primary_paper",
            support_summary="Evaluate candidate basis terms with granular marginal generalization influence.",
            limitation_summary="A recent preprint cannot supply local validity or real-world discovery guarantees.",
        ),
        SourceRecordV313.seal(
            source_id="prov_o_2013",
            title="PROV-O The PROV Ontology W3C Recommendation",
            persistent_id="W3C-REC-prov-o-20130430",
            canonical_url="https://www.w3.org/TR/prov-o/",
            source_tier="normative_standard",
            support_summary="Represent derivation generation attribution and invalidation as explicit provenance.",
            limitation_summary="Provenance structure does not prove source truth or scientific validity.",
        ),
        SourceRecordV313.seal(
            source_id="sbml_l3v2r2",
            title="SBML Level 3 Version 2 Core Release 2",
            persistent_id="doi:10.1515/jib-2019-0021",
            canonical_url="https://sbml.org/documents/specifications/level-3/version-2/core/",
            source_tier="normative_standard",
            support_summary="Use structured mathematical objects and explicit unit semantics for model interchange.",
            limitation_summary="This local DSL is not a complete SBML implementation or conformance claim.",
        ),
        SourceRecordV313.seal(
            source_id="pysr_official",
            title="PySR official operator and constraint documentation",
            persistent_id="github:MilesCranmer/PySR",
            canonical_url="https://ai.damtp.cam.ac.uk/pysr/options/",
            source_tier="official_software",
            support_summary="Constrain operators nesting arguments depth and expression complexity.",
            limitation_summary="FMA forbids the custom executable operator path supported by PySR.",
        ),
        SourceRecordV313.seal(
            source_id="richards_1959",
            title="A Flexible Growth Function for Empirical Use",
            persistent_id="doi:10.1093/jxb/10.2.290",
            canonical_url="https://cir.nii.ac.jp/crid/1360855569998926848?lang=en",
            source_tier="primary_paper",
            support_summary="Provides the historical flexible generalized growth-function family used in the fixture.",
            limitation_summary="An empirical growth family is not universally causal or domain-valid.",
        ),
        SourceRecordV313.seal(
            source_id="monod_test_1971",
            title="Control of Growth Rate by Initial Substrate Concentration at Values Below Maximum Rate",
            persistent_id="doi:10.1128/am.22.6.1041-1047.1971",
            canonical_url="https://journals.asm.org/doi/10.1128/am.22.6.1041-1047.1971",
            source_tier="primary_paper",
            support_summary="Tests the hyperbolic Monod relationship between specific growth rate and substrate.",
            limitation_summary="The paper reports that the instantaneous relationship was not observed in its batch experiment.",
        ),
        SourceRecordV313.seal(
            source_id="v312_private_admission",
            title="FMA V3.12 private open-set concept admission",
            persistent_id=f"sha256:{v312_report_hash}",
            canonical_url="https://local.invalid/fma/v312/private-admission",
            source_tier="prior_private_admission",
            support_summary="Privately admitted logarithmic capacity growth in the sealed synthetic V3.12 worldpack.",
            limitation_summary="Synthetic admission is not a bibliographic claim or real-world mechanism approval.",
        ),
    ]
    source_by_id = {item.source_id: item for item in sources}

    def claim(
        claim_id: str,
        source_id: str,
        kind: ClaimKindV313,
        locator: str,
        summary: str,
        contradictions: list[str] | None = None,
    ) -> ScientificClaimV313:
        return ScientificClaimV313.seal(
            claim_id=claim_id,
            source_record_hash=source_by_id[source_id].record_hash,
            claim_kind=kind,
            locator=locator,
            claim_summary=summary,
            contradiction_claim_hashes=contradictions or [],
        )

    claims = [
        claim(
            "claim_llm_sr_skeleton_parameter_split", "llm_sr_2025",
            "compiler_constraint", "abstract",
            "Represent a proposed skeleton separately from deterministic parameter fitting.",
        ),
        claim(
            "claim_lasr_concept_library", "lasr_2024",
            "compiler_constraint", "abstract",
            "Keep reusable concepts separate from the current hypothesis population.",
        ),
        claim(
            "claim_igsr_term_influence", "igsr_2026",
            "compiler_constraint", "abstract",
            "Evaluate a term by its marginal contribution to generalization accuracy.",
        ),
        claim(
            "claim_prov_derivation_invalidation", "prov_o_2013",
            "compiler_constraint", "sections 2-3",
            "Record derivation generation and invalidation in an explicit provenance chain.",
        ),
        claim(
            "claim_sbml_units", "sbml_l3v2r2",
            "unit_relation", "core specification",
            "Mathematical model objects require explicit time state and parameter unit semantics.",
        ),
        claim(
            "claim_pysr_operator_constraints", "pysr_official",
            "compiler_constraint", "operator constraints",
            "Restrict operator arguments nesting and expression complexity before search.",
        ),
        claim(
            "claim_richards_generalized_growth", "richards_1959",
            "operator_template", "growth function family",
            "Use a dimensionless powered state-to-capacity ratio in a flexible growth law.",
        ),
        claim(
            "claim_monod_hyperbolic_rate", "monod_test_1971",
            "operator_template", "abstract hyperbolic relationship",
            "Use a hyperbolic saturation ratio for a substrate-conditioned growth rate.",
        ),
        claim(
            "claim_monod_instantaneous_limit", "monod_test_1971",
            "known_limitation", "abstract negative result",
            "Do not assume the instantaneous Monod relationship held in the reported batch experiment.",
        ),
        claim(
            "claim_v312_log_capacity", "v312_private_admission",
            "operator_template", "sealed V3.12 concept ledger",
            "Carry logarithmic state-to-capacity growth only as a prior synthetic concept candidate.",
        ),
    ]
    # Bind the support claim to its explicit negative scope claim without a
    # circular claim hash: recreate the support claim after the limitation hash.
    limitation = next(
        item for item in claims if item.claim_id == "claim_monod_instantaneous_limit"
    )
    claims = [
        ScientificClaimV313.seal(
            **item.model_dump(exclude={"claim_hash", "contradiction_claim_hashes"}),
            contradiction_claim_hashes=[limitation.claim_hash],
        ) if item.claim_id == "claim_monod_hyperbolic_rate" else item
        for item in claims
    ]
    return ConceptEvidenceBundleV313.seal(
        evidence_id="evidence_to_concept_compiler_sources_v313",
        sources=sources,
        claims=claims,
    )


def default_concept_packages_v313(
    evidence: ConceptEvidenceBundleV313,
) -> list[ConceptPackageV313]:
    evidence.assert_sealed()
    claim_hash = {item.claim_id: item.claim_hash for item in evidence.claims}
    state = _state()
    gompertz_rhs = _operator(
        "multiply",
        _operator("multiply", _parameter("r"), state),
        _operator("log", _operator("divide", _parameter("K"), state)),
    )
    richards_rhs = _operator(
        "multiply",
        _operator("multiply", _parameter("r"), state),
        _operator(
            "subtract",
            _constant(1.0),
            _operator(
                "power",
                _operator("divide", state, _parameter("K")),
                _parameter("nu"),
            ),
        ),
    )
    monod_rhs = _operator(
        "subtract",
        _operator(
            "divide",
            _operator("multiply", _parameter("a"), state),
            _operator("add", _parameter("b"), state),
        ),
        _operator("multiply", _parameter("c"), state),
    )
    affine_decoy_rhs = _operator(
        "subtract",
        _operator("multiply", _parameter("r"), state),
        _parameter("q"),
    )
    return [
        ConceptPackageV313.seal(
            concept_id="log_capacity_growth",
            evidence_hash=evidence.evidence_hash,
            supporting_claim_hashes=[claim_hash["claim_v312_log_capacity"]],
            rhs=gompertz_rhs,
            parameters=[
                ParameterSpecV313(
                    parameter_id="r", dimension=DimensionV313(time_power=-1),
                    lower_bound=0.02, upper_bound=2.0, initial_value=0.4,
                ),
                ParameterSpecV313(
                    parameter_id="K", dimension=STATE_DIMENSION_V313,
                    lower_bound=0.2, upper_bound=20.0, initial_value=3.0,
                ),
            ],
            state_domain_lower=0.01,
            state_domain_upper=20.0,
        ),
        ConceptPackageV313.seal(
            concept_id="generalized_capacity_growth",
            evidence_hash=evidence.evidence_hash,
            supporting_claim_hashes=[claim_hash["claim_richards_generalized_growth"]],
            rhs=richards_rhs,
            parameters=[
                ParameterSpecV313(
                    parameter_id="r", dimension=DimensionV313(time_power=-1),
                    lower_bound=0.02, upper_bound=2.0, initial_value=0.4,
                ),
                ParameterSpecV313(
                    parameter_id="K", dimension=STATE_DIMENSION_V313,
                    lower_bound=0.2, upper_bound=20.0, initial_value=3.0,
                ),
                ParameterSpecV313(
                    parameter_id="nu", dimension=DIMENSIONLESS_V313,
                    lower_bound=0.2, upper_bound=4.0, initial_value=1.3,
                ),
            ],
            state_domain_lower=0.01,
            state_domain_upper=20.0,
        ),
        ConceptPackageV313.seal(
            concept_id="hyperbolic_net_growth",
            evidence_hash=evidence.evidence_hash,
            supporting_claim_hashes=[
                claim_hash["claim_monod_hyperbolic_rate"],
                claim_hash["claim_monod_instantaneous_limit"],
            ],
            rhs=monod_rhs,
            parameters=[
                ParameterSpecV313(
                    parameter_id="a", dimension=RHS_DIMENSION_V313,
                    lower_bound=0.05, upper_bound=10.0, initial_value=1.0,
                ),
                ParameterSpecV313(
                    parameter_id="b", dimension=STATE_DIMENSION_V313,
                    lower_bound=0.02, upper_bound=10.0, initial_value=1.0,
                ),
                ParameterSpecV313(
                    parameter_id="c", dimension=DimensionV313(time_power=-1),
                    lower_bound=0.001, upper_bound=1.0, initial_value=0.1,
                ),
            ],
            state_domain_lower=0.01,
            state_domain_upper=20.0,
        ),
        ConceptPackageV313.seal(
            concept_id="affine_rate_decoy",
            evidence_hash=evidence.evidence_hash,
            supporting_claim_hashes=[claim_hash["claim_pysr_operator_constraints"]],
            rhs=affine_decoy_rhs,
            parameters=[
                ParameterSpecV313(
                    parameter_id="r", dimension=DimensionV313(time_power=-1),
                    lower_bound=-2.0, upper_bound=2.0, initial_value=0.1,
                ),
                ParameterSpecV313(
                    parameter_id="q", dimension=RHS_DIMENSION_V313,
                    lower_bound=-5.0, upper_bound=5.0, initial_value=0.1,
                ),
            ],
            state_domain_lower=0.01,
            state_domain_upper=20.0,
        ),
    ]
