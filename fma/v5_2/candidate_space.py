"""Strict generated-candidate language and code-owned admission for V5.2."""

from __future__ import annotations

import hashlib
import hmac
import math
import re
from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import Field, model_validator

from fma.hashing import sha256_value
from fma.schemas import StrictModel
from fma.v2.schemas import Identifier, Sha256, _assert_timezone


ExpressionOpV52 = Literal[
    "observable",
    "state",
    "parameter",
    "constant",
    "add",
    "subtract",
    "multiply",
    "divide",
    "power",
    "exp",
    "log",
    "sum_history",
    "derivative",
    "output",
]
AdmissionStatusV52 = Literal["ADMITTED", "REJECTED"]


def _hash_without(model: StrictModel, *fields: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude=set(fields)))


def _clean_dimension(value: dict[str, int]) -> dict[str, int]:
    return dict(sorted((key, exponent) for key, exponent in value.items() if exponent))


def _dimension_add(
    first: dict[str, int], second: dict[str, int], factor: int = 1
) -> dict[str, int]:
    output = dict(first)
    for key, exponent in second.items():
        output[key] = output.get(key, 0) + factor * exponent
    return _clean_dimension(output)


class ExpressionNodeV52(StrictModel):
    node_id: Identifier
    op: ExpressionOpV52
    inputs: Annotated[list[Identifier], Field(max_length=8)] = Field(
        default_factory=list
    )
    dimension: dict[Identifier, Annotated[int, Field(ge=-12, le=12)]] = Field(
        default_factory=dict
    )
    semantic_role: Identifier | None = None
    constant_value: Annotated[float, Field(allow_inf_nan=False)] | None = None
    lower_bound: Annotated[float, Field(allow_inf_nan=False)] | None = None
    upper_bound: Annotated[float, Field(allow_inf_nan=False)] | None = None
    exponent: Annotated[int, Field(ge=-8, le=8)] | None = None

    @model_validator(mode="after")
    def validate_node(self) -> "ExpressionNodeV52":
        if any(value == 0 for value in self.dimension.values()):
            raise ValueError("dimension vector must omit zero exponents")
        leaf = self.op in {"observable", "state", "parameter", "constant"}
        if leaf != (not self.inputs):
            raise ValueError("only leaf nodes may omit inputs")
        if self.op in {"observable", "state", "parameter"} and not self.semantic_role:
            raise ValueError("named leaves require semantic_role")
        if self.op == "constant":
            if self.constant_value is None:
                raise ValueError("constant node needs constant_value")
        elif self.constant_value is not None:
            raise ValueError("only constant nodes may set constant_value")
        if self.op == "parameter":
            if (
                self.lower_bound is not None
                and self.upper_bound is not None
                and self.lower_bound >= self.upper_bound
            ):
                raise ValueError("parameter lower bound must be below upper bound")
        elif self.lower_bound is not None or self.upper_bound is not None:
            raise ValueError("only parameter nodes may set bounds")
        if self.op == "power":
            if self.exponent is None or self.exponent == 0:
                raise ValueError("power node needs a nonzero integer exponent")
        elif self.exponent is not None:
            raise ValueError("only power nodes may set exponent")
        expected_arity = {
            "add": 2,
            "subtract": 2,
            "multiply": 2,
            "divide": 2,
            "power": 1,
            "exp": 1,
            "log": 1,
            "sum_history": 1,
            "derivative": 1,
            "output": 1,
        }
        if self.op in expected_arity and len(self.inputs) != expected_arity[self.op]:
            raise ValueError(f"{self.op} has the wrong arity")
        return self


class LimitCaseV52(StrictModel):
    parameter_node_id: Identifier
    limit_value: Annotated[float, Field(allow_inf_nan=False)]
    reduces_to_family: Identifier
    executable_check_id: Identifier


class IdentifiabilityObligationV52(StrictModel):
    obligation_id: Identifier
    parameter_node_ids: Annotated[list[Identifier], Field(min_length=1)]
    executable_check_id: Identifier
    failure_consequence: Annotated[str, Field(min_length=5, max_length=1000)]

    @model_validator(mode="after")
    def validate_obligation(self) -> "IdentifiabilityObligationV52":
        if self.parameter_node_ids != sorted(set(self.parameter_node_ids)):
            raise ValueError("identifiability parameter IDs must be sorted and unique")
        return self


class GeneratedCandidateV52(StrictModel):
    """A model proposal in a mechanically checkable equation-DAG language."""

    schema_version: Literal["5.2"] = "5.2"
    candidate_id: Identifier
    domain_id: Identifier
    family: Identifier
    generation: Annotated[int, Field(ge=0, le=32)]
    nodes: Annotated[list[ExpressionNodeV52], Field(min_length=2, max_length=256)]
    output_node_id: Identifier
    assumptions: Annotated[
        list[Annotated[str, Field(min_length=5, max_length=1000)]],
        Field(min_length=1, max_length=32),
    ]
    data_requirements: Annotated[
        list[Identifier], Field(min_length=1, max_length=32)
    ]
    limit_cases: Annotated[list[LimitCaseV52], Field(min_length=1, max_length=32)]
    identifiability_obligations: Annotated[
        list[IdentifiabilityObligationV52], Field(min_length=1, max_length=32)
    ]
    expected_failure_modes: Annotated[
        list[Annotated[str, Field(min_length=5, max_length=1000)]],
        Field(min_length=1, max_length=32),
    ]
    parent_candidate_hashes: list[Sha256] = Field(default_factory=list)
    operator_ids: list[Identifier] = Field(default_factory=list)
    generator_process_receipt_hash: Sha256
    private_evidence_used: Literal[False] = False
    candidate_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_candidate(self) -> "GeneratedCandidateV52":
        node_ids = [node.node_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("candidate node IDs must be unique")
        known: dict[str, ExpressionNodeV52] = {}
        for node in self.nodes:
            if any(source not in known for source in node.inputs):
                raise ValueError("equation graph is not topologically ordered")
            expected = self._expected_dimension(node, known)
            if _clean_dimension(node.dimension) != expected:
                raise ValueError(
                    f"node {node.node_id} dimension differs from operator algebra"
                )
            known[node.node_id] = node
        if self.output_node_id not in known:
            raise ValueError("output node is missing")
        if known[self.output_node_id].op != "output":
            raise ValueError("output_node_id must identify an output node")

        parameters = {
            node.node_id for node in self.nodes if node.op == "parameter"
        }
        if not parameters:
            raise ValueError("generated candidate needs at least one parameter")
        for limit in self.limit_cases:
            if limit.parameter_node_id not in parameters:
                raise ValueError("limit case references a non-parameter node")
        covered: set[str] = set()
        obligation_ids: list[str] = []
        for obligation in self.identifiability_obligations:
            obligation_ids.append(obligation.obligation_id)
            if not set(obligation.parameter_node_ids).issubset(parameters):
                raise ValueError(
                    "identifiability obligation references a non-parameter"
                )
            covered.update(obligation.parameter_node_ids)
        if obligation_ids != sorted(set(obligation_ids)):
            raise ValueError("identifiability obligation IDs must be sorted and unique")
        if covered != parameters:
            raise ValueError("identifiability obligations must cover every parameter")
        for label, values in (
            ("data_requirements", self.data_requirements),
            ("parent_candidate_hashes", self.parent_candidate_hashes),
            ("operator_ids", self.operator_ids),
        ):
            if values != sorted(set(values)):
                raise ValueError(f"{label} must be sorted and unique")
        if self.generation == 0 and (
            self.parent_candidate_hashes or self.operator_ids
        ):
            raise ValueError("generation-zero candidate cannot claim lineage")
        if self.generation > 0 and (
            not self.parent_candidate_hashes or not self.operator_ids
        ):
            raise ValueError("evolved candidate requires parent and operator lineage")
        if self.candidate_hash and self.candidate_hash != self.content_hash():
            raise ValueError("candidate_hash does not match generated candidate")
        return self

    @staticmethod
    def _expected_dimension(
        node: ExpressionNodeV52,
        known: dict[str, ExpressionNodeV52],
    ) -> dict[str, int]:
        if node.op in {"observable", "state", "parameter", "constant"}:
            return _clean_dimension(node.dimension)
        inputs = [known[item].dimension for item in node.inputs]
        if node.op in {"add", "subtract"}:
            if _clean_dimension(inputs[0]) != _clean_dimension(inputs[1]):
                raise ValueError("add/subtract inputs have different dimensions")
            return _clean_dimension(inputs[0])
        if node.op == "multiply":
            return _dimension_add(inputs[0], inputs[1])
        if node.op == "divide":
            return _dimension_add(inputs[0], inputs[1], -1)
        if node.op == "power":
            assert node.exponent is not None
            return {
                key: value * node.exponent for key, value in inputs[0].items()
            }
        if node.op in {"exp", "log"}:
            if _clean_dimension(inputs[0]):
                raise ValueError("exp/log input must be dimensionless")
            return {}
        if node.op in {"sum_history", "output"}:
            return _clean_dimension(inputs[0])
        if node.op == "derivative":
            return _dimension_add(inputs[0], {"time": 1}, -1)
        raise AssertionError("unhandled expression operator")

    def content_hash(self) -> str:
        return _hash_without(self, "candidate_hash")

    @classmethod
    def seal(cls, **data: object) -> "GeneratedCandidateV52":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"candidate_hash"}),
            candidate_hash=draft.content_hash(),
        )

    def assert_sealed(self) -> None:
        if not self.candidate_hash or self.candidate_hash != self.content_hash():
            raise ValueError("generated candidate is not sealed")

    def structural_signature(self) -> str:
        signatures: dict[str, object] = {}
        commutative = {"add", "multiply"}
        for node in self.nodes:
            inputs = [signatures[item] for item in node.inputs]
            if node.op in commutative:
                inputs = sorted(inputs, key=sha256_value)
            payload: dict[str, object] = {
                "op": node.op,
                "inputs": inputs,
                "dimension": _clean_dimension(node.dimension),
            }
            if node.op in {"state", "observable"}:
                payload["semantic_role"] = node.semantic_role
            if node.op == "parameter":
                payload["bounds"] = [node.lower_bound, node.upper_bound]
            if node.op == "constant":
                payload["constant_value"] = node.constant_value
            if node.op == "power":
                payload["exponent"] = node.exponent
            signatures[node.node_id] = payload
        return sha256_value(
            {
                "domain_id": self.domain_id,
                "root": signatures[self.output_node_id],
            }
        )


class CandidateAdmissionPolicyV52(StrictModel):
    schema_version: Literal["5.2"] = "5.2"
    policy_id: Identifier
    allowed_domain_ids: Annotated[list[Identifier], Field(min_length=1)]
    allowed_operators: Annotated[list[ExpressionOpV52], Field(min_length=1)]
    available_check_ids: Annotated[list[Identifier], Field(min_length=1)]
    required_baseline_candidate_hashes: list[Sha256] = Field(default_factory=list)
    max_candidates: Annotated[int, Field(ge=1, le=512)] = 32
    max_nodes_per_candidate: Annotated[int, Field(ge=2, le=256)] = 64
    max_parameters_per_candidate: Annotated[int, Field(ge=1, le=64)] = 12
    require_parameter_bounds: bool = True
    require_executable_limit_case: bool = True
    require_identifiability: bool = True
    policy_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_policy(self) -> "CandidateAdmissionPolicyV52":
        for label, values in (
            ("allowed_domain_ids", self.allowed_domain_ids),
            ("allowed_operators", self.allowed_operators),
            ("available_check_ids", self.available_check_ids),
            (
                "required_baseline_candidate_hashes",
                self.required_baseline_candidate_hashes,
            ),
        ):
            if values != sorted(set(values)):
                raise ValueError(f"{label} must be sorted and unique")
        if self.policy_hash and self.policy_hash != self.content_hash():
            raise ValueError("policy_hash does not match admission policy")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "policy_hash")

    @classmethod
    def seal(cls, **data: object) -> "CandidateAdmissionPolicyV52":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"policy_hash"}),
            policy_hash=draft.content_hash(),
        )

    def assert_sealed(self) -> None:
        if not self.policy_hash or self.policy_hash != self.content_hash():
            raise ValueError("candidate admission policy is not sealed")


class CandidateAdmissionReceiptV52(StrictModel):
    schema_version: Literal["5.2"] = "5.2"
    admission_id: Identifier
    candidate_hash: Sha256
    structural_signature: Sha256
    generator_process_receipt_hash: Sha256
    policy_hash: Sha256
    registry_before_hash: Sha256
    registry_after_hash: Sha256
    status: AdmissionStatusV52
    checks: dict[Identifier, bool]
    reasons: list[Identifier]
    authority_key_id: Identifier
    admitted_at: datetime
    private_evidence_used: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    authority_auth_tag: Sha256 | None = None
    receipt_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_receipt(self) -> "CandidateAdmissionReceiptV52":
        _assert_timezone(self.admitted_at, "admitted_at")
        if self.reasons != sorted(set(self.reasons)):
            raise ValueError("admission reasons must be sorted and unique")
        admitted = self.status == "ADMITTED"
        if admitted != (all(self.checks.values()) and not self.reasons):
            raise ValueError("admission status disagrees with checks")
        if admitted == (self.registry_before_hash == self.registry_after_hash):
            raise ValueError("registry hash transition disagrees with admission")
        if self.authority_auth_tag and not self.receipt_hash:
            raise ValueError("authenticated admission needs receipt_hash")
        if self.receipt_hash and self.receipt_hash != self.content_hash():
            raise ValueError("receipt_hash does not match candidate admission")
        return self

    def unsigned_hash(self) -> str:
        return _hash_without(self, "authority_auth_tag", "receipt_hash")

    def content_hash(self) -> str:
        return _hash_without(self, "receipt_hash")


class CandidateAdmissionAuthorityV52:
    def __init__(self, key_id: str, secret: bytes) -> None:
        if not re.fullmatch(r"^[A-Za-z][A-Za-z0-9_.-]*$", key_id):
            raise ValueError("invalid candidate admission key_id")
        if len(secret) < 32:
            raise ValueError("candidate admission secret needs at least 32 bytes")
        self.key_id = key_id
        self._secret = bytes(secret)

    def _mac(self, unsigned_hash: str) -> str:
        return hmac.new(
            self._secret,
            f"candidate_admission_v52:{unsigned_hash}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def issue(self, **data: object) -> CandidateAdmissionReceiptV52:
        data["authority_key_id"] = self.key_id
        data.setdefault("admitted_at", datetime.now(timezone.utc))
        unsigned = CandidateAdmissionReceiptV52(**data)
        final_payload = unsigned.model_dump(mode="json")
        final_payload["authority_auth_tag"] = self._mac(
            unsigned.unsigned_hash()
        )
        final_payload["receipt_hash"] = sha256_value(
            {
                key: value
                for key, value in final_payload.items()
                if key != "receipt_hash"
            }
        )
        return CandidateAdmissionReceiptV52(**final_payload)

    def verify(self, receipt: CandidateAdmissionReceiptV52) -> bool:
        try:
            return bool(
                receipt.receipt_hash
                and receipt.receipt_hash == receipt.content_hash()
                and receipt.authority_key_id == self.key_id
                and receipt.authority_auth_tag
                and hmac.compare_digest(
                    receipt.authority_auth_tag,
                    self._mac(receipt.unsigned_hash()),
                )
            )
        except (TypeError, ValueError):
            return False


class GovernedCandidateRegistryV52:
    """In-memory admission state whose every transition is receipt-bound."""

    def __init__(
        self,
        *,
        policy: CandidateAdmissionPolicyV52,
        authority: CandidateAdmissionAuthorityV52,
        initial_candidates: list[GeneratedCandidateV52] | None = None,
    ) -> None:
        policy.assert_sealed()
        self.policy = policy
        self.authority = authority
        self._candidates: list[GeneratedCandidateV52] = []
        self._receipts: list[CandidateAdmissionReceiptV52] = []
        for candidate in initial_candidates or []:
            candidate.assert_sealed()
            self._candidates.append(candidate)
        if not set(policy.required_baseline_candidate_hashes).issubset(
            {item.candidate_hash for item in self._candidates}
        ):
            raise ValueError("initial registry does not contain required baselines")

    @property
    def candidates(self) -> list[GeneratedCandidateV52]:
        return list(self._candidates)

    @property
    def receipts(self) -> list[CandidateAdmissionReceiptV52]:
        return list(self._receipts)

    def registry_hash(
        self, candidates: list[GeneratedCandidateV52] | None = None
    ) -> str:
        values = candidates if candidates is not None else self._candidates
        return sha256_value(
            [
                {
                    "candidate_hash": item.candidate_hash,
                    "structural_signature": item.structural_signature(),
                }
                for item in sorted(values, key=lambda item: str(item.candidate_hash))
            ]
        )

    def admit(
        self,
        candidate: GeneratedCandidateV52,
        *,
        observed_generator_receipt_hashes: set[str],
    ) -> CandidateAdmissionReceiptV52:
        candidate.assert_sealed()
        before = self.registry_hash()
        existing_hashes = {item.candidate_hash for item in self._candidates}
        existing_signatures = {
            item.structural_signature() for item in self._candidates
        }
        parameter_nodes = [
            item for item in candidate.nodes if item.op == "parameter"
        ]
        used_ops = {item.op for item in candidate.nodes}
        executable_checks = {
            item.executable_check_id for item in candidate.limit_cases
        } | {
            item.executable_check_id
            for item in candidate.identifiability_obligations
        }
        all_parameters_bounded = all(
            item.lower_bound is not None and item.upper_bound is not None
            for item in parameter_nodes
        )
        checks = {
            "baseline_registry_preserved": set(
                self.policy.required_baseline_candidate_hashes
            ).issubset(existing_hashes),
            "candidate_budget_available": len(self._candidates)
            < self.policy.max_candidates,
            "candidate_hash_new": candidate.candidate_hash not in existing_hashes,
            "domain_allowed": candidate.domain_id
            in self.policy.allowed_domain_ids,
            "generator_receipt_observed": (
                candidate.generator_process_receipt_hash
                in observed_generator_receipt_hashes
            ),
            "identifiability_declared": (
                bool(candidate.identifiability_obligations)
                or not self.policy.require_identifiability
            ),
            "limit_case_executable": (
                bool(candidate.limit_cases)
                or not self.policy.require_executable_limit_case
            ),
            "node_budget": len(candidate.nodes)
            <= self.policy.max_nodes_per_candidate,
            "operator_allowlist": used_ops.issubset(
                set(self.policy.allowed_operators)
            ),
            "parameter_bounds": (
                all_parameters_bounded or not self.policy.require_parameter_bounds
            ),
            "parameter_budget": len(parameter_nodes)
            <= self.policy.max_parameters_per_candidate,
            "scientific_checks_available": executable_checks.issubset(
                set(self.policy.available_check_ids)
            ),
            "structurally_novel": candidate.structural_signature()
            not in existing_signatures,
        }
        reasons = sorted(key for key, passed in checks.items() if not passed)
        admitted = not reasons
        after_candidates = (
            [*self._candidates, candidate] if admitted else self._candidates
        )
        after = self.registry_hash(after_candidates)
        receipt = self.authority.issue(
            admission_id=f"admission.{candidate.candidate_id}",
            candidate_hash=candidate.candidate_hash,
            structural_signature=candidate.structural_signature(),
            generator_process_receipt_hash=(
                candidate.generator_process_receipt_hash
            ),
            policy_hash=self.policy.policy_hash,
            registry_before_hash=before,
            registry_after_hash=after,
            status="ADMITTED" if admitted else "REJECTED",
            checks=dict(sorted(checks.items())),
            reasons=reasons,
        )
        if not self.authority.verify(receipt):
            raise RuntimeError("candidate admission receipt failed authentication")
        self._receipts.append(receipt)
        if admitted:
            self._candidates.append(candidate)
        return receipt


__all__ = [
    "CandidateAdmissionAuthorityV52",
    "CandidateAdmissionPolicyV52",
    "CandidateAdmissionReceiptV52",
    "ExpressionNodeV52",
    "GeneratedCandidateV52",
    "GovernedCandidateRegistryV52",
    "IdentifiabilityObligationV52",
    "LimitCaseV52",
]
