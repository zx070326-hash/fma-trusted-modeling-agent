"""Shared executable conformance kit for code-owned V6.8 capability packs.

The objects in this module are a development-only runtime boundary.  A
``CapabilityRuntimeDefinitionV68`` contains direct Python callable references;
entry-point strings in artifacts are never imported or executed.  Every
definition is bound to one exact sealed capability manifest.

The conformance report proves only that local fixtures exercised the declared
compiler, executor, verifier, rejection, and benchmark-case contracts.  It is
not L0--L4 evidence for a real task, a stage gate, or scientific qualification.
"""

from __future__ import annotations

import hashlib
import inspect
import marshal
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Mapping

from pydantic import Field, model_validator

from fma.hashing import sha256_value
from fma.schemas import StrictModel
from fma.v2.schemas import Identifier, Sha256

from .capability_sdk_v68 import (
    CapabilityConformanceReportV68,
    CapabilityManifestV68,
    LevelV68,
    evaluate_capability_conformance_v68,
)


_LEVELS: tuple[LevelV68, ...] = ("L0", "L1", "L2", "L3", "L4")

RuntimeCaseModeV68 = Literal[
    "compile_execute",
    "tamper_rejection",
    "incompatible_rejection",
    "not_run",
]
RuntimeObservedOutcomeV68 = Literal[
    "EXECUTED",
    "REJECTED",
    "NOT_RUN",
    "ERROR",
]
RuntimeContractStatusV68 = Literal["PASS", "FAIL", "NOT_RUN"]
BenchmarkCoverageStatusV68 = Literal["PASS", "FAIL", "NOT_RUN"]


def _hash_without(model: StrictModel, *fields: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude=set(fields)))


def callable_semantic_hash_v68(
    callable_object: Callable[..., object],
) -> str:
    """Return a local semantic identity without resolving an artifact string."""

    code = getattr(callable_object, "__code__", None)
    source_path = inspect.getsourcefile(callable_object)
    if code is None or source_path is None:
        raise ValueError("V6.8 runtime received an unhashable callable")
    path = Path(source_path).resolve()
    return sha256_value(
        {
            "module": callable_object.__module__,
            "qualname": callable_object.__qualname__,
            "marshalled_code_sha256": hashlib.sha256(
                marshal.dumps(code)
            ).hexdigest(),
            "module_source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "defaults": repr(getattr(callable_object, "__defaults__", None)),
            "keyword_defaults": repr(
                getattr(callable_object, "__kwdefaults__", None)
            ),
        }
    )


class CapabilityRuntimeInvocationV68(StrictModel):
    """Sealed, non-executable input envelope for one code-owned fixture."""

    schema_version: Literal["6.8-capability-runtime-invocation"] = (
        "6.8-capability-runtime-invocation"
    )
    manifest_id: Identifier
    manifest_hash: Sha256
    case_id: Identifier
    payload_schema_id: Identifier
    payload_schema_hash: Sha256
    payload: dict[str, Any]
    fixture_only: Literal[True] = True
    payload_is_executable: Literal[False] = False
    arbitrary_code_execution_permitted: Literal[False] = False
    invocation_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_invocation(self) -> "CapabilityRuntimeInvocationV68":
        if self.invocation_hash and self.invocation_hash != self.content_hash():
            raise ValueError("V6.8 runtime invocation hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "invocation_hash")

    @classmethod
    def seal(cls, **data: object) -> "CapabilityRuntimeInvocationV68":
        draft = cls(**data)
        return cls(
            **draft.model_dump(mode="json", exclude={"invocation_hash"}),
            invocation_hash=draft.content_hash(),
        )

    def assert_sealed(self) -> None:
        if (
            not self.invocation_hash
            or self.invocation_hash != self.content_hash()
        ):
            raise ValueError("V6.8 runtime invocation is not sealed")


class CapabilityRuntimeExecutionV68(StrictModel):
    """Sealed domain payload returned by a code-owned runtime adapter."""

    schema_version: Literal["6.8-capability-runtime-execution"] = (
        "6.8-capability-runtime-execution"
    )
    manifest_id: Identifier
    manifest_hash: Sha256
    case_id: Identifier
    payload_schema_id: Identifier
    payload_schema_hash: Sha256
    payload: dict[str, Any]
    fixture_only: Literal[True] = True
    execution_is_scientific_evidence: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    execution_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_execution(self) -> "CapabilityRuntimeExecutionV68":
        if self.execution_hash and self.execution_hash != self.content_hash():
            raise ValueError("V6.8 runtime execution hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "execution_hash")

    @classmethod
    def seal(cls, **data: object) -> "CapabilityRuntimeExecutionV68":
        draft = cls(**data)
        return cls(
            **draft.model_dump(mode="json", exclude={"execution_hash"}),
            execution_hash=draft.content_hash(),
        )

    def assert_sealed(self) -> None:
        if not self.execution_hash or self.execution_hash != self.content_hash():
            raise ValueError("V6.8 runtime execution is not sealed")


class CapabilityRuntimeCaseContractV68(StrictModel):
    """Frozen expected behavior for one public local conformance fixture."""

    schema_version: Literal["6.8-capability-runtime-case-contract"] = (
        "6.8-capability-runtime-case-contract"
    )
    case_id: Identifier
    case_kind: Identifier
    mode: RuntimeCaseModeV68
    expected_outcome: RuntimeObservedOutcomeV68
    expected_level_statuses: dict[LevelV68, Identifier] = Field(
        default_factory=dict
    )
    expected_exception_types: list[Identifier] = Field(default_factory=list)
    deterministic_execution_required: bool
    public_benchmark_case: bool
    adversarial_case: bool
    fixture_only: Literal[True] = True
    not_run_counts_as_scientific_pass: Literal[False] = False
    case_contract_is_scientific_evidence: Literal[False] = False
    contract_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_contract(self) -> "CapabilityRuntimeCaseContractV68":
        expected_by_mode = {
            "compile_execute": "EXECUTED",
            "tamper_rejection": "REJECTED",
            "incompatible_rejection": "REJECTED",
            "not_run": "NOT_RUN",
        }
        if self.expected_outcome != expected_by_mode[self.mode]:
            raise ValueError("V6.8 runtime case outcome differs from mode")
        if self.expected_level_statuses and list(
            self.expected_level_statuses
        ) != list(_LEVELS):
            raise ValueError(
                "V6.8 expected level statuses must contain ordered L0-L4"
            )
        if self.mode == "compile_execute":
            if self.expected_exception_types:
                raise ValueError(
                    "executed V6.8 case cannot expect an exception"
                )
        elif self.expected_level_statuses:
            raise ValueError(
                "non-executed V6.8 case cannot expect level statuses"
            )
        if self.mode in {"tamper_rejection", "incompatible_rejection"}:
            if not self.expected_exception_types:
                raise ValueError("rejection case requires exception types")
        elif self.expected_exception_types:
            raise ValueError(
                "non-rejection V6.8 case cannot expect exception types"
            )
        if self.deterministic_execution_required and self.mode != (
            "compile_execute"
        ):
            raise ValueError(
                "only an executed V6.8 case can require determinism"
            )
        if self.adversarial_case and not self.public_benchmark_case:
            raise ValueError(
                "V6.8 adversarial case must be a public benchmark case"
            )
        if self.expected_exception_types != sorted(
            set(self.expected_exception_types)
        ):
            raise ValueError("V6.8 expected exception types must be sorted")
        if self.contract_hash and self.contract_hash != self.content_hash():
            raise ValueError("V6.8 runtime case contract hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "contract_hash")

    @classmethod
    def seal(cls, **data: object) -> "CapabilityRuntimeCaseContractV68":
        draft = cls(**data)
        return cls(
            **draft.model_dump(mode="json", exclude={"contract_hash"}),
            contract_hash=draft.content_hash(),
        )

    def assert_sealed(self) -> None:
        if not self.contract_hash or self.contract_hash != self.content_hash():
            raise ValueError("V6.8 runtime case contract is not sealed")


RuntimeInputFactoryV68 = Callable[[], CapabilityRuntimeInvocationV68]
RuntimeCompilerV68 = Callable[
    [CapabilityRuntimeInvocationV68],
    StrictModel,
]
RuntimeExecutorV68 = Callable[
    [CapabilityRuntimeInvocationV68, StrictModel],
    CapabilityRuntimeExecutionV68,
]
RuntimeLevelVerifierV68 = Callable[
    [
        CapabilityRuntimeInvocationV68,
        StrictModel,
        CapabilityRuntimeExecutionV68,
        LevelV68,
    ],
    StrictModel,
]
RuntimeIRMutatorV68 = Callable[[StrictModel], StrictModel]
RuntimeProbeV68 = Callable[[], None]
RuntimeOutputAssertionV68 = Callable[
    [CapabilityRuntimeExecutionV68],
    None,
]


@dataclass(frozen=True)
class CapabilityRuntimeCaseBindingV68:
    """Code-owned functions for one sealed case contract."""

    contract: CapabilityRuntimeCaseContractV68
    input_factory: RuntimeInputFactoryV68 | None = None
    ir_mutator: RuntimeIRMutatorV68 | None = None
    incompatible_probe: RuntimeProbeV68 | None = None
    output_assertion: RuntimeOutputAssertionV68 | None = None

    def __post_init__(self) -> None:
        self.contract.assert_sealed()
        if self.contract.mode == "compile_execute":
            if self.input_factory is None:
                raise ValueError("executed V6.8 case needs an input factory")
            if self.ir_mutator is not None or self.incompatible_probe is not None:
                raise ValueError("executed V6.8 case has incompatible callbacks")
        elif self.contract.mode == "tamper_rejection":
            if self.input_factory is None or self.ir_mutator is None:
                raise ValueError("tamper V6.8 case needs input and mutator")
            if self.incompatible_probe is not None:
                raise ValueError("tamper V6.8 case cannot have a probe")
        elif self.contract.mode == "incompatible_rejection":
            if self.incompatible_probe is None:
                raise ValueError("incompatible V6.8 case needs a probe")
            if (
                self.input_factory is not None
                or self.ir_mutator is not None
                or self.output_assertion is not None
            ):
                raise ValueError(
                    "incompatible V6.8 case has execution callbacks"
                )
        elif any(
            item is not None
            for item in (
                self.input_factory,
                self.ir_mutator,
                self.incompatible_probe,
                self.output_assertion,
            )
        ):
            raise ValueError("NOT_RUN V6.8 case cannot have callbacks")

    def binding_hash(self) -> str:
        callbacks = {
            "input_factory": self.input_factory,
            "ir_mutator": self.ir_mutator,
            "incompatible_probe": self.incompatible_probe,
            "output_assertion": self.output_assertion,
        }
        return sha256_value(
            {
                "contract_hash": self.contract.contract_hash,
                "callbacks": {
                    key: (
                        callable_semantic_hash_v68(value)
                        if value is not None
                        else None
                    )
                    for key, value in callbacks.items()
                },
            }
        )


@dataclass(frozen=True)
class CapabilityRuntimeDefinitionV68:
    """Exact-manifest runtime definition containing direct callable references."""

    definition_id: str
    manifest: CapabilityManifestV68
    observed_semantic_hashes: Mapping[str, str]
    benchmark_spec: Mapping[str, object]
    cases: tuple[CapabilityRuntimeCaseBindingV68, ...]
    compiler: RuntimeCompilerV68 | None
    executor: RuntimeExecutorV68 | None
    level_verifier: RuntimeLevelVerifierV68 | None
    unavailable_reason: str | None = None
    dynamic_import_permitted: Literal[False] = False
    model_supplied_callable_permitted: Literal[False] = False
    definition_hash: str = field(init=False)

    def __post_init__(self) -> None:
        self.manifest.assert_sealed()
        if not self.definition_id:
            raise ValueError("V6.8 runtime definition needs an ID")
        if set(self.observed_semantic_hashes) != set(
            self.manifest.expected_semantic_hashes()
        ):
            raise ValueError(
                "V6.8 runtime observed identity set differs from manifest"
            )
        if sha256_value(dict(self.benchmark_spec)) != (
            self.manifest.benchmark.benchmark_suite_hash
        ):
            raise ValueError(
                "V6.8 runtime benchmark spec differs from manifest"
            )
        if self.benchmark_spec.get("minimum_public_cases") != (
            self.manifest.benchmark.minimum_public_cases
        ):
            raise ValueError("V6.8 benchmark public minimum differs")
        if self.benchmark_spec.get("minimum_adversarial_cases") != (
            self.manifest.benchmark.minimum_adversarial_cases
        ):
            raise ValueError("V6.8 benchmark adversarial minimum differs")
        case_ids = [item.contract.case_id for item in self.cases]
        if case_ids != sorted(set(case_ids)):
            raise ValueError("V6.8 runtime cases must be sorted and unique")
        callbacks = (self.compiler, self.executor, self.level_verifier)
        if self.unavailable_reason is None:
            if any(item is None for item in callbacks):
                raise ValueError(
                    "available V6.8 runtime needs compiler, executor, verifier"
                )
        elif any(item is not None for item in callbacks):
            raise ValueError(
                "unavailable V6.8 runtime cannot expose partial callbacks"
            )
        payload = {
            "schema_version": "6.8-capability-runtime-definition",
            "definition_id": self.definition_id,
            "manifest_id": self.manifest.manifest_id,
            "manifest_hash": self.manifest.manifest_hash,
            "observed_semantic_hashes": dict(
                sorted(self.observed_semantic_hashes.items())
            ),
            "benchmark_spec_hash": sha256_value(dict(self.benchmark_spec)),
            "case_binding_hashes": [
                item.binding_hash() for item in self.cases
            ],
            "compiler_adapter_hash": (
                callable_semantic_hash_v68(self.compiler)
                if self.compiler is not None
                else None
            ),
            "executor_adapter_hash": (
                callable_semantic_hash_v68(self.executor)
                if self.executor is not None
                else None
            ),
            "verifier_adapter_hash": (
                callable_semantic_hash_v68(self.level_verifier)
                if self.level_verifier is not None
                else None
            ),
            "unavailable_reason": self.unavailable_reason,
            "dynamic_import_permitted": False,
            "model_supplied_callable_permitted": False,
        }
        object.__setattr__(self, "definition_hash", sha256_value(payload))

    @property
    def runtime_available(self) -> bool:
        return self.unavailable_reason is None

    def assert_exact_manifest(
        self,
        *,
        manifest_id: str,
        manifest_hash: str,
    ) -> None:
        if (
            manifest_id != self.manifest.manifest_id
            or manifest_hash != self.manifest.manifest_hash
        ):
            raise KeyError("V6.8 runtime definition manifest hash mismatch")
        self.manifest.assert_sealed()


class CapabilityRuntimeCaseReceiptV68(StrictModel):
    """Sealed receipt for one local fixture case."""

    schema_version: Literal["6.8-capability-runtime-case-receipt"] = (
        "6.8-capability-runtime-case-receipt"
    )
    definition_id: Identifier
    definition_hash: Sha256
    manifest_id: Identifier
    manifest_hash: Sha256
    case_id: Identifier
    case_kind: Identifier
    case_contract_hash: Sha256
    mode: RuntimeCaseModeV68
    expected_outcome: RuntimeObservedOutcomeV68
    observed_outcome: RuntimeObservedOutcomeV68
    expected_level_statuses: dict[LevelV68, Identifier]
    observed_level_statuses: dict[LevelV68, Identifier]
    expected_exception_types: list[Identifier]
    observed_exception_type: Identifier | None
    invocation_hash: Sha256 | None
    typed_ir_hash: Sha256 | None
    first_execution_hash: Sha256 | None
    second_execution_hash: Sha256 | None
    verifier_evidence_hashes: dict[LevelV68, Sha256]
    deterministic_execution_required: bool
    deterministic_execution_confirmed: bool | None
    public_benchmark_case: bool
    adversarial_case: bool
    contract_status: RuntimeContractStatusV68
    reason_code: Identifier
    error_fingerprint: Sha256 | None = None
    fixture_only: Literal[True] = True
    receipt_is_scientific_evidence: Literal[False] = False
    scientific_evidence_status: Literal["NOT_RUN"] = "NOT_RUN"
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    receipt_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_receipt(self) -> "CapabilityRuntimeCaseReceiptV68":
        if self.expected_level_statuses and list(
            self.expected_level_statuses
        ) != list(_LEVELS):
            raise ValueError("V6.8 receipt expected levels differ")
        if self.observed_outcome == "EXECUTED":
            if list(self.observed_level_statuses) != list(_LEVELS):
                raise ValueError("executed V6.8 receipt needs ordered L0-L4")
            if list(self.verifier_evidence_hashes) != list(_LEVELS):
                raise ValueError(
                    "executed V6.8 receipt needs ordered verifier hashes"
                )
            if not all(
                (
                    self.invocation_hash,
                    self.typed_ir_hash,
                    self.first_execution_hash,
                )
            ):
                raise ValueError("executed V6.8 receipt lacks artifact hashes")
        elif self.observed_level_statuses or self.verifier_evidence_hashes:
            raise ValueError("non-executed V6.8 receipt contains levels")
        if self.expected_exception_types != sorted(
            set(self.expected_exception_types)
        ):
            raise ValueError("V6.8 receipt exception types must be sorted")
        expected_status: RuntimeContractStatusV68
        if self.observed_outcome == "NOT_RUN":
            expected_status = (
                "NOT_RUN"
                if self.expected_outcome == "NOT_RUN"
                else "FAIL"
            )
        elif self.observed_outcome != self.expected_outcome:
            expected_status = "FAIL"
        elif self.observed_outcome == "REJECTED":
            expected_status = (
                "PASS"
                if self.observed_exception_type
                in self.expected_exception_types
                else "FAIL"
            )
        elif self.observed_outcome == "EXECUTED":
            deterministic_ok = (
                not self.deterministic_execution_required
                or self.deterministic_execution_confirmed is True
            )
            levels_ok = (
                not self.expected_level_statuses
                or self.expected_level_statuses
                == self.observed_level_statuses
            )
            expected_status = (
                "PASS" if deterministic_ok and levels_ok else "FAIL"
            )
        else:
            expected_status = "FAIL"
        if self.contract_status != expected_status:
            raise ValueError("V6.8 case contract status differs")
        expected_reason = {
            "PASS": (
                "runtime_case_rejected_as_expected"
                if self.observed_outcome == "REJECTED"
                else "runtime_case_executed_as_expected"
            ),
            "FAIL": "runtime_case_contract_mismatch",
            "NOT_RUN": "runtime_case_not_run",
        }[self.contract_status]
        if self.reason_code != expected_reason:
            raise ValueError("V6.8 runtime case reason differs")
        if self.receipt_hash and self.receipt_hash != self.content_hash():
            raise ValueError("V6.8 runtime case receipt hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "receipt_hash")

    @classmethod
    def seal(cls, **data: object) -> "CapabilityRuntimeCaseReceiptV68":
        draft = cls(**data)
        return cls(
            **draft.model_dump(mode="json", exclude={"receipt_hash"}),
            receipt_hash=draft.content_hash(),
        )

    def assert_sealed(self) -> None:
        if not self.receipt_hash or self.receipt_hash != self.content_hash():
            raise ValueError("V6.8 runtime case receipt is not sealed")


class CapabilityRuntimeConformanceReportV68(StrictModel):
    """Local contract report; never scientific or stage authority."""

    schema_version: Literal["6.8-capability-runtime-conformance-report"] = (
        "6.8-capability-runtime-conformance-report"
    )
    definition_id: Identifier
    definition_hash: Sha256
    manifest_id: Identifier
    manifest_hash: Sha256
    identity_conformance_status: Literal["PASS", "FAIL"]
    identity_conformance_report_hash: Sha256
    receipts: list[CapabilityRuntimeCaseReceiptV68]
    expected_benchmark_case_ids: list[Identifier]
    observed_benchmark_case_ids: list[Identifier]
    observed_public_case_count: int
    observed_adversarial_case_count: int
    minimum_public_case_count: int
    minimum_adversarial_case_count: int
    benchmark_coverage_status: BenchmarkCoverageStatusV68
    status: RuntimeContractStatusV68
    fixture_only: Literal[True] = True
    report_is_scientific_evidence: Literal[False] = False
    scientific_evidence_status: Literal["NOT_RUN"] = "NOT_RUN"
    maturity_promotion_granted: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    report_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_report(self) -> "CapabilityRuntimeConformanceReportV68":
        receipt_ids = [item.case_id for item in self.receipts]
        if receipt_ids != sorted(set(receipt_ids)):
            raise ValueError("V6.8 runtime receipts must be sorted and unique")
        for receipt in self.receipts:
            receipt.assert_sealed()
            if (
                receipt.definition_id != self.definition_id
                or receipt.definition_hash != self.definition_hash
                or receipt.manifest_id != self.manifest_id
                or receipt.manifest_hash != self.manifest_hash
            ):
                raise ValueError("V6.8 runtime receipt binding differs")
        expected_coverage: BenchmarkCoverageStatusV68
        if not self.expected_benchmark_case_ids:
            expected_coverage = "NOT_RUN"
        elif (
            self.expected_benchmark_case_ids
            != self.observed_benchmark_case_ids
            or self.observed_public_case_count
            < self.minimum_public_case_count
            or self.observed_adversarial_case_count
            < self.minimum_adversarial_case_count
        ):
            expected_coverage = "FAIL"
        else:
            expected_coverage = "PASS"
        if self.benchmark_coverage_status != expected_coverage:
            raise ValueError("V6.8 benchmark coverage status differs")
        if (
            self.identity_conformance_status == "FAIL"
            or self.benchmark_coverage_status == "FAIL"
            or any(item.contract_status == "FAIL" for item in self.receipts)
        ):
            expected_status: RuntimeContractStatusV68 = "FAIL"
        elif (
            self.benchmark_coverage_status == "NOT_RUN"
            or any(item.contract_status == "NOT_RUN" for item in self.receipts)
        ):
            expected_status = "NOT_RUN"
        else:
            expected_status = "PASS"
        if self.status != expected_status:
            raise ValueError("V6.8 runtime conformance status differs")
        if self.report_hash and self.report_hash != self.content_hash():
            raise ValueError("V6.8 runtime conformance report hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "report_hash")

    @classmethod
    def seal(cls, **data: object) -> "CapabilityRuntimeConformanceReportV68":
        draft = cls(**data)
        return cls(
            **draft.model_dump(mode="json", exclude={"report_hash"}),
            report_hash=draft.content_hash(),
        )

    def assert_sealed(self) -> None:
        if not self.report_hash or self.report_hash != self.content_hash():
            raise ValueError("V6.8 runtime conformance report is not sealed")


def _assert_sealed_if_supported(value: StrictModel) -> None:
    assert_method = getattr(value, "assert_sealed", None)
    if callable(assert_method):
        assert_method()


def _strict_model_hash(value: StrictModel) -> str:
    return sha256_value(value.model_dump(mode="json"))


def _status_text(value: object) -> str:
    raw = getattr(value, "value", value)
    return str(raw)


def _receipt_status(
    *,
    contract: CapabilityRuntimeCaseContractV68,
    observed_outcome: RuntimeObservedOutcomeV68,
    observed_exception_type: str | None,
    observed_levels: Mapping[str, str],
    deterministic_confirmed: bool | None,
) -> RuntimeContractStatusV68:
    if observed_outcome == "NOT_RUN":
        return "NOT_RUN" if contract.expected_outcome == "NOT_RUN" else "FAIL"
    if observed_outcome != contract.expected_outcome:
        return "FAIL"
    if observed_outcome == "REJECTED":
        return (
            "PASS"
            if observed_exception_type in contract.expected_exception_types
            else "FAIL"
        )
    if observed_outcome != "EXECUTED":
        return "FAIL"
    if (
        contract.deterministic_execution_required
        and deterministic_confirmed is not True
    ):
        return "FAIL"
    if (
        contract.expected_level_statuses
        and dict(contract.expected_level_statuses) != dict(observed_levels)
    ):
        return "FAIL"
    return "PASS"


def _run_case_v68(
    definition: CapabilityRuntimeDefinitionV68,
    binding: CapabilityRuntimeCaseBindingV68,
) -> CapabilityRuntimeCaseReceiptV68:
    contract = binding.contract
    observed_outcome: RuntimeObservedOutcomeV68 = "ERROR"
    invocation_hash: str | None = None
    typed_ir_hash: str | None = None
    first_execution_hash: str | None = None
    second_execution_hash: str | None = None
    observed_levels: dict[str, str] = {}
    verifier_hashes: dict[str, str] = {}
    observed_exception_type: str | None = None
    deterministic_confirmed: bool | None = None
    error_fingerprint: str | None = None

    if contract.mode == "not_run":
        observed_outcome = "NOT_RUN"
    elif contract.mode == "incompatible_rejection":
        try:
            if binding.incompatible_probe is None:
                raise RuntimeError("missing incompatible probe")
            binding.incompatible_probe()
            observed_outcome = "ERROR"
        except Exception as exc:  # noqa: BLE001 - the receipt records the type
            observed_outcome = "REJECTED"
            observed_exception_type = type(exc).__name__
            error_fingerprint = sha256_value(
                {
                    "exception_type": observed_exception_type,
                    "message": str(exc),
                }
            )
    else:
        try:
            if (
                binding.input_factory is None
                or definition.compiler is None
                or definition.executor is None
            ):
                raise RuntimeError("V6.8 runtime callbacks are unavailable")
            invocation = binding.input_factory()
            invocation.assert_sealed()
            definition.assert_exact_manifest(
                manifest_id=invocation.manifest_id,
                manifest_hash=invocation.manifest_hash,
            )
            if invocation.case_id != contract.case_id:
                raise ValueError("V6.8 runtime invocation case differs")
            invocation_hash = str(invocation.invocation_hash)
            model_ir = definition.compiler(invocation)
            if not isinstance(model_ir, StrictModel):
                raise TypeError("V6.8 compiler did not return StrictModel")
            _assert_sealed_if_supported(model_ir)
            typed_ir_hash = _strict_model_hash(model_ir)
            if contract.mode == "tamper_rejection":
                if binding.ir_mutator is None:
                    raise RuntimeError("V6.8 tamper mutator is absent")
                tampered_ir = binding.ir_mutator(model_ir)
                if not isinstance(tampered_ir, StrictModel):
                    raise TypeError("V6.8 tamper mutator returned invalid IR")
                try:
                    definition.executor(invocation, tampered_ir)
                    observed_outcome = "EXECUTED"
                except Exception as exc:  # noqa: BLE001
                    observed_outcome = "REJECTED"
                    observed_exception_type = type(exc).__name__
                    error_fingerprint = sha256_value(
                        {
                            "exception_type": observed_exception_type,
                            "message": str(exc),
                        }
                    )
            else:
                first = definition.executor(invocation, model_ir)
                first.assert_sealed()
                if (
                    first.manifest_id != definition.manifest.manifest_id
                    or first.manifest_hash != definition.manifest.manifest_hash
                    or first.case_id != contract.case_id
                ):
                    raise ValueError("V6.8 execution binding differs")
                if binding.output_assertion is not None:
                    binding.output_assertion(first)
                first_execution_hash = str(first.execution_hash)
                if contract.deterministic_execution_required:
                    second = definition.executor(invocation, model_ir)
                    second.assert_sealed()
                    second_execution_hash = str(second.execution_hash)
                    deterministic_confirmed = first == second
                if definition.level_verifier is None:
                    raise RuntimeError("V6.8 level verifier is unavailable")
                for level in _LEVELS:
                    evidence = definition.level_verifier(
                        invocation,
                        model_ir,
                        first,
                        level,
                    )
                    if not isinstance(evidence, StrictModel):
                        raise TypeError(
                            "V6.8 verifier did not return StrictModel"
                        )
                    _assert_sealed_if_supported(evidence)
                    if _status_text(getattr(evidence, "level", None)) != level:
                        raise ValueError("V6.8 verifier level differs")
                    observed_levels[level] = _status_text(
                        getattr(evidence, "status", None)
                    )
                    verifier_hashes[level] = _strict_model_hash(evidence)
                observed_outcome = "EXECUTED"
        except Exception as exc:  # noqa: BLE001 - fail-closed receipt
            if observed_outcome != "REJECTED":
                observed_outcome = "ERROR"
                observed_exception_type = type(exc).__name__
                error_fingerprint = sha256_value(
                    {
                        "exception_type": observed_exception_type,
                        "message": str(exc),
                    }
                )

    contract_status = _receipt_status(
        contract=contract,
        observed_outcome=observed_outcome,
        observed_exception_type=observed_exception_type,
        observed_levels=observed_levels,
        deterministic_confirmed=deterministic_confirmed,
    )
    reason_code = {
        "PASS": (
            "runtime_case_rejected_as_expected"
            if observed_outcome == "REJECTED"
            else "runtime_case_executed_as_expected"
        ),
        "FAIL": "runtime_case_contract_mismatch",
        "NOT_RUN": "runtime_case_not_run",
    }[contract_status]
    return CapabilityRuntimeCaseReceiptV68.seal(
        definition_id=definition.definition_id,
        definition_hash=definition.definition_hash,
        manifest_id=definition.manifest.manifest_id,
        manifest_hash=definition.manifest.manifest_hash,
        case_id=contract.case_id,
        case_kind=contract.case_kind,
        case_contract_hash=contract.contract_hash,
        mode=contract.mode,
        expected_outcome=contract.expected_outcome,
        observed_outcome=observed_outcome,
        expected_level_statuses=contract.expected_level_statuses,
        observed_level_statuses=observed_levels,
        expected_exception_types=contract.expected_exception_types,
        observed_exception_type=observed_exception_type,
        invocation_hash=invocation_hash,
        typed_ir_hash=typed_ir_hash,
        first_execution_hash=first_execution_hash,
        second_execution_hash=second_execution_hash,
        verifier_evidence_hashes=verifier_hashes,
        deterministic_execution_required=(
            contract.deterministic_execution_required
        ),
        deterministic_execution_confirmed=deterministic_confirmed,
        public_benchmark_case=contract.public_benchmark_case,
        adversarial_case=contract.adversarial_case,
        contract_status=contract_status,
        reason_code=reason_code,
        error_fingerprint=error_fingerprint,
    )


def _identity_report_v68(
    definition: CapabilityRuntimeDefinitionV68,
) -> CapabilityConformanceReportV68:
    return evaluate_capability_conformance_v68(
        definition.manifest,
        observed_semantic_hashes=definition.observed_semantic_hashes,
    )


def run_capability_runtime_conformance_v68(
    definition: CapabilityRuntimeDefinitionV68,
) -> CapabilityRuntimeConformanceReportV68:
    """Run the shared local kit against one exact code-owned definition."""

    definition.assert_exact_manifest(
        manifest_id=definition.manifest.manifest_id,
        manifest_hash=str(definition.manifest.manifest_hash),
    )
    identity = _identity_report_v68(definition)
    identity.assert_sealed()
    receipts = sorted(
        (
            _run_case_v68(definition, binding)
            for binding in definition.cases
        ),
        key=lambda item: item.case_id,
    )
    expected_case_ids_raw = definition.benchmark_spec.get("case_ids")
    expected_case_ids = (
        sorted(str(item) for item in expected_case_ids_raw)
        if isinstance(expected_case_ids_raw, list)
        else []
    )
    public_receipts = [
        item for item in receipts if item.public_benchmark_case
    ]
    observed_case_ids = sorted(item.case_id for item in public_receipts)
    observed_adversarial = sum(
        1 for item in public_receipts if item.adversarial_case
    )
    minimum_public = definition.manifest.benchmark.minimum_public_cases
    minimum_adversarial = (
        definition.manifest.benchmark.minimum_adversarial_cases
    )
    if not expected_case_ids:
        coverage_status: BenchmarkCoverageStatusV68 = "NOT_RUN"
    elif (
        expected_case_ids != observed_case_ids
        or len(public_receipts) < minimum_public
        or observed_adversarial < minimum_adversarial
    ):
        coverage_status = "FAIL"
    else:
        coverage_status = "PASS"
    if (
        identity.status == "FAIL"
        or coverage_status == "FAIL"
        or any(item.contract_status == "FAIL" for item in receipts)
    ):
        status: RuntimeContractStatusV68 = "FAIL"
    elif coverage_status == "NOT_RUN" or any(
        item.contract_status == "NOT_RUN" for item in receipts
    ):
        status = "NOT_RUN"
    else:
        status = "PASS"
    return CapabilityRuntimeConformanceReportV68.seal(
        definition_id=definition.definition_id,
        definition_hash=definition.definition_hash,
        manifest_id=definition.manifest.manifest_id,
        manifest_hash=definition.manifest.manifest_hash,
        identity_conformance_status=identity.status,
        identity_conformance_report_hash=identity.report_hash,
        receipts=receipts,
        expected_benchmark_case_ids=expected_case_ids,
        observed_benchmark_case_ids=observed_case_ids,
        observed_public_case_count=len(public_receipts),
        observed_adversarial_case_count=observed_adversarial,
        minimum_public_case_count=minimum_public,
        minimum_adversarial_case_count=minimum_adversarial,
        benchmark_coverage_status=coverage_status,
        status=status,
    )


__all__ = [
    "CapabilityRuntimeCaseBindingV68",
    "CapabilityRuntimeCaseContractV68",
    "CapabilityRuntimeCaseReceiptV68",
    "CapabilityRuntimeConformanceReportV68",
    "CapabilityRuntimeDefinitionV68",
    "CapabilityRuntimeExecutionV68",
    "CapabilityRuntimeInvocationV68",
    "callable_semantic_hash_v68",
    "run_capability_runtime_conformance_v68",
]
