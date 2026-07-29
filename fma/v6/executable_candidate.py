"""V6.2 executable semantics for the registered positive-series adapters.

The S1 candidate prose is documentation, not executable code.  This module
turns the selected candidate into a narrow registered-family-search IR, binds
the concrete adapter and its exact family registry to the current S2 attempt,
and checks the resulting S3 scientific bundle before issuing a receipt.

The receipt is content-sealed but is not standalone authority.  It must be
admitted by the normal authenticated stage manifest.  Local execution,
including a fully passing fixture bundle, cannot grant scientific
qualification or authorize a real-world action.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypeAlias, cast

from pydantic import Field, model_validator

from fma.hashing import sha256_value
from fma.schemas import StrictModel
from fma.v2.schemas import Identifier, Sha256
from fma.v5.workspace_schemas import (
    CandidateFormalizationV50,
    CandidateSetV50,
    ModelSpecV50,
)
from fma.v5_2.ode_system import ODEScientificBundleV52
from fma.v5_7.adaptive_positive_series import (
    AdaptivePositiveSeriesBundleV57,
)


EXECUTABLE_CANDIDATE_INTENT_PATH = (
    "docs/executable_candidate_intent_v62.json"
)
EXECUTABLE_CANDIDATE_IR_PATH = "docs/executable_candidate_ir_v62.json"
EXECUTABLE_CANDIDATE_RESOLUTION_PATH = (
    "docs/executable_candidate_resolution_v62.json"
)
EXECUTABLE_CANDIDATE_RECEIPT_PATH = (
    "results/executable_candidate_receipt_v62.json"
)

SCALAR_ODE_ADAPTER_ID = "scalar_autonomous_ode_v52"
ADAPTIVE_POSITIVE_SERIES_ADAPTER_ID = "adaptive_positive_series_v57"

AdapterIdV62 = Literal[
    "scalar_autonomous_ode_v52",
    "adaptive_positive_series_v57",
]
RegisteredFamilyV62 = Literal[
    "constant",
    "exponential",
    "gompertz",
    "logistic",
    "log_random_walk_drift",
    "log_growth_ar1",
]
ScientificBundleV62: TypeAlias = (
    ODEScientificBundleV52 | AdaptivePositiveSeriesBundleV57
)

SUPPORTED_ADAPTER_IDS_V62: tuple[AdapterIdV62, ...] = (
    SCALAR_ODE_ADAPTER_ID,
    ADAPTIVE_POSITIVE_SERIES_ADAPTER_ID,
)
SCALAR_ODE_FAMILIES_V62: tuple[RegisteredFamilyV62, ...] = (
    "constant",
    "exponential",
    "gompertz",
    "logistic",
)
ADAPTIVE_POSITIVE_SERIES_FAMILIES_V62: tuple[
    RegisteredFamilyV62, ...
] = (
    *SCALAR_ODE_FAMILIES_V62,
    "log_random_walk_drift",
    "log_growth_ar1",
)

_ODE_EQUATIONS_V52 = {
    "constant": "dx/dt = 0",
    "exponential": "dx/dt = r*x",
    "gompertz": "dx/dt = r*x*log(K/x)",
    "logistic": "dx/dt = r*x*(1-x/K)",
}
_HYBRID_RESIDUAL_MODES_V56 = ("trend_only", "ar1_residual")
_HYBRID_GRAPH_POLICY_V56 = (
    "trend_only_then_triggered_ar1_then_mechanism_guard"
)


class ExecutableCandidateError(ValueError):
    """Raised when an executable-candidate binding fails closed."""


def _hash_without(model: StrictModel, *fields: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude=set(fields)))


def registered_families_for_adapter_v62(
    adapter_id: AdapterIdV62,
) -> tuple[RegisteredFamilyV62, ...]:
    if adapter_id == SCALAR_ODE_ADAPTER_ID:
        return SCALAR_ODE_FAMILIES_V62
    if adapter_id == ADAPTIVE_POSITIVE_SERIES_ADAPTER_ID:
        return ADAPTIVE_POSITIVE_SERIES_FAMILIES_V62
    raise ExecutableCandidateError(f"unsupported execution adapter: {adapter_id}")


def allowed_family_registry_hash_v62(
    adapter_id: AdapterIdV62,
) -> str:
    return sha256_value(
        {
            "schema_version": "6.2-allowed-family-registry",
            "adapter_id": adapter_id,
            "families": list(registered_families_for_adapter_v62(adapter_id)),
        }
    )


class RegisteredFamilySearchIntentV62(StrictModel):
    """Model-owned S1 typed intent; it contains no executable prose."""

    schema_version: Literal["6.2-registered-family-search-intent"] = (
        "6.2-registered-family-search-intent"
    )
    candidate_id: Identifier
    operation: Literal["registered_family_search"] = "registered_family_search"
    input_domain: Literal["positive_scalar_time_series"] = (
        "positive_scalar_time_series"
    )
    allowed_adapter_ids: Annotated[
        list[AdapterIdV62], Field(min_length=2, max_length=2)
    ]
    adapter_resolution_stage: Literal["S2"] = "S2"
    model_family_text_executable: Literal[False] = False
    mathematical_form_text_executable: Literal[False] = False
    arbitrary_code_execution_permitted: Literal[False] = False

    @model_validator(mode="after")
    def validate_intent(self) -> "RegisteredFamilySearchIntentV62":
        if self.allowed_adapter_ids != list(SUPPORTED_ADAPTER_IDS_V62):
            raise ValueError(
                "S1 intent must expose exactly the registered positive-series "
                "adapters"
            )
        return self

    def content_hash(self) -> str:
        return sha256_value(self.model_dump(mode="json"))


class RegisteredFamilySearchIRV62(StrictModel):
    """S1 executable IR; candidate prose is deliberately not represented."""

    schema_version: Literal["6.2-registered-family-search-ir"] = (
        "6.2-registered-family-search-ir"
    )
    candidate_id: Identifier
    candidate_structural_hash: Sha256
    model_intent_hash: Sha256
    operation: Literal["registered_family_search"] = "registered_family_search"
    input_domain: Literal["positive_scalar_time_series"] = (
        "positive_scalar_time_series"
    )
    allowed_adapter_ids: Annotated[
        list[AdapterIdV62], Field(min_length=2, max_length=2)
    ]
    adapter_resolution_stage: Literal["S2"] = "S2"
    model_family_text_executable: Literal[False] = False
    mathematical_form_text_executable: Literal[False] = False
    arbitrary_code_execution_permitted: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    ir_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_ir(self) -> "RegisteredFamilySearchIRV62":
        if self.allowed_adapter_ids != list(SUPPORTED_ADAPTER_IDS_V62):
            raise ValueError(
                "S1 IR must expose exactly the registered positive-series "
                "adapters"
            )
        if self.ir_hash and self.ir_hash != self.content_hash():
            raise ValueError("executable-candidate IR hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "ir_hash")

    def assert_sealed(self) -> None:
        if not self.ir_hash or self.ir_hash != self.content_hash():
            raise ExecutableCandidateError(
                "executable-candidate IR is not sealed"
            )

    @classmethod
    def seal(cls, **data: object) -> "RegisteredFamilySearchIRV62":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"ir_hash"})
        payload["ir_hash"] = draft.content_hash()
        return cls(**payload)


class ExecutableCandidateResolutionV62(StrictModel):
    """Harness-owned S2 resolution of an S1 search IR to one adapter."""

    schema_version: Literal["6.2-executable-candidate-resolution"] = (
        "6.2-executable-candidate-resolution"
    )
    workspace_spec_hash: Sha256
    s1_gate_hash: Sha256
    s2_attempt: Annotated[int, Field(ge=1)]
    execution_ir_hash: Sha256
    model_spec_hash: Sha256
    selected_candidate_id: Identifier
    selected_candidate_structural_hash: Sha256
    adapter_id: AdapterIdV62
    allowed_families: Annotated[
        list[RegisteredFamilyV62], Field(min_length=4, max_length=6)
    ]
    allowed_family_registry_hash: Sha256
    resolution_scope: Literal["current_s2_attempt"] = "current_s2_attempt"
    resolved_by: Literal["harness"] = "harness"
    free_text_execution_permitted: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    resolution_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_resolution(self) -> "ExecutableCandidateResolutionV62":
        expected = list(registered_families_for_adapter_v62(self.adapter_id))
        if self.allowed_families != expected:
            raise ValueError(
                "S2 resolution differs from the exact registered families"
            )
        if self.allowed_family_registry_hash != (
            allowed_family_registry_hash_v62(self.adapter_id)
        ):
            raise ValueError("S2 allowed-family registry hash differs")
        if (
            self.resolution_hash
            and self.resolution_hash != self.content_hash()
        ):
            raise ValueError("executable-candidate resolution hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "resolution_hash")

    def assert_sealed(self) -> None:
        if (
            not self.resolution_hash
            or self.resolution_hash != self.content_hash()
        ):
            raise ExecutableCandidateError(
                "executable-candidate resolution is not sealed"
            )

    @classmethod
    def seal(cls, **data: object) -> "ExecutableCandidateResolutionV62":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"resolution_hash"})
        payload["resolution_hash"] = draft.content_hash()
        return cls(**payload)


class ExecutableCandidateReceiptV62(StrictModel):
    """S3 receipt for the exact registered search observed in a bundle."""

    schema_version: Literal["6.2-executable-candidate-receipt"] = (
        "6.2-executable-candidate-receipt"
    )
    workspace_spec_hash: Sha256
    s1_gate_hash: Sha256
    s2_gate_hash: Sha256
    s2_attempt: Annotated[int, Field(ge=1)]
    resolution_hash: Sha256
    selected_candidate_structural_hash: Sha256
    adapter_id: AdapterIdV62
    allowed_families: Annotated[
        list[RegisteredFamilyV62], Field(min_length=4, max_length=6)
    ]
    allowed_family_registry_hash: Sha256
    evaluated_families: Annotated[
        list[RegisteredFamilyV62], Field(min_length=1)
    ]
    evaluated_model_ids: Annotated[list[Identifier], Field(min_length=1)]
    selected_family: RegisteredFamilyV62
    selected_model_id: Identifier
    bundle_schema_version: Literal["5.2", "5.7"]
    bundle_task_id: Identifier
    bundle_hash: Sha256
    candidate_registry_hash: Sha256
    candidate_graph_hash: Sha256 | None = None
    nested_candidate_graph_hash: Sha256 | None = None
    bundle_scientific_acceptance: bool
    fixture_only: bool
    local_execution_status: Literal["PASS"] = "PASS"
    scientific_qualification_status: Literal["NOT_RUN"] = "NOT_RUN"
    receipt_authority: Literal["authenticated_s3_manifest_required"] = (
        "authenticated_s3_manifest_required"
    )
    receipt_is_standalone_authority: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    receipt_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_receipt(self) -> "ExecutableCandidateReceiptV62":
        expected = list(registered_families_for_adapter_v62(self.adapter_id))
        if self.allowed_families != expected:
            raise ValueError(
                "execution receipt differs from the registered families"
            )
        if self.allowed_family_registry_hash != (
            allowed_family_registry_hash_v62(self.adapter_id)
        ):
            raise ValueError("execution receipt registry hash differs")
        if self.evaluated_families != sorted(set(self.evaluated_families)):
            raise ValueError(
                "evaluated families must be sorted and unique"
            )
        if not set(self.evaluated_families).issubset(
            set(self.allowed_families)
        ):
            raise ValueError("bundle evaluated an unregistered family")
        if self.selected_family not in self.evaluated_families:
            raise ValueError("selected family was not evaluated")
        if self.evaluated_model_ids != sorted(set(self.evaluated_model_ids)):
            raise ValueError(
                "evaluated model IDs must be sorted and unique"
            )
        if self.selected_model_id not in self.evaluated_model_ids:
            raise ValueError("selected model was not evaluated")
        if self.adapter_id == SCALAR_ODE_ADAPTER_ID:
            if (
                self.bundle_schema_version != "5.2"
                or self.candidate_graph_hash is not None
                or self.nested_candidate_graph_hash is not None
            ):
                raise ValueError(
                    "scalar ODE receipt contains an adaptive graph binding"
                )
        elif (
            self.bundle_schema_version != "5.7"
            or self.candidate_graph_hash is None
            or self.nested_candidate_graph_hash is None
        ):
            raise ValueError(
                "adaptive receipt lacks its nested graph bindings"
            )
        if self.receipt_hash and self.receipt_hash != self.content_hash():
            raise ValueError("executable-candidate receipt hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "receipt_hash")

    def assert_sealed(self) -> None:
        if not self.receipt_hash or self.receipt_hash != self.content_hash():
            raise ExecutableCandidateError(
                "executable-candidate receipt is not sealed"
            )

    @classmethod
    def seal(cls, **data: object) -> "ExecutableCandidateReceiptV62":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"receipt_hash"})
        payload["receipt_hash"] = draft.content_hash()
        return cls(**payload)


def compile_registered_family_search_ir_v62(
    candidate: CandidateFormalizationV50,
    intent: RegisteredFamilySearchIntentV62,
) -> RegisteredFamilySearchIRV62:
    """Bind a model-owned typed intent to candidate identity and structure."""

    if intent.candidate_id != candidate.candidate_id:
        raise ExecutableCandidateError(
            "registered-family intent belongs to another candidate"
        )
    return RegisteredFamilySearchIRV62.seal(
        candidate_id=candidate.candidate_id,
        candidate_structural_hash=candidate.structural_hash(),
        model_intent_hash=intent.content_hash(),
        operation=intent.operation,
        input_domain=intent.input_domain,
        allowed_adapter_ids=intent.allowed_adapter_ids,
        adapter_resolution_stage=intent.adapter_resolution_stage,
        model_family_text_executable=intent.model_family_text_executable,
        mathematical_form_text_executable=(
            intent.mathematical_form_text_executable
        ),
        arbitrary_code_execution_permitted=(
            intent.arbitrary_code_execution_permitted
        ),
    )


def _sealed_model_spec(model_spec: ModelSpecV50) -> None:
    if (
        not model_spec.model_hash
        or model_spec.model_hash != model_spec.content_hash()
    ):
        raise ExecutableCandidateError("selected model spec is not sealed")


def _workspace_spec_hash(workspace: Any) -> str:
    spec_hash = getattr(getattr(workspace, "spec", None), "spec_hash", None)
    if not isinstance(spec_hash, str):
        raise ExecutableCandidateError("workspace spec is not sealed")
    return spec_hash


def _workspace_s2_attempt(workspace: Any) -> int:
    try:
        attempt = workspace._latest_attempt("S2")
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ExecutableCandidateError(
            "current S2 attempt cannot be resolved"
        ) from exc
    if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 1:
        raise ExecutableCandidateError("current S2 attempt is invalid")
    return attempt


def _assert_resolution_current(
    workspace: Any,
    resolution: ExecutableCandidateResolutionV62,
) -> None:
    resolution.assert_sealed()
    if resolution.workspace_spec_hash != _workspace_spec_hash(workspace):
        raise ExecutableCandidateError(
            "resolution belongs to another workspace"
        )
    if workspace.current_gate("S1") != resolution.s1_gate_hash:
        raise ExecutableCandidateError(
            "resolution belongs to a stale S1 gate"
        )
    if _workspace_s2_attempt(workspace) != resolution.s2_attempt:
        raise ExecutableCandidateError(
            "resolution belongs to a stale S2 attempt"
        )


def resolve_executable_candidate_v62(
    *,
    workspace: Any,
    execution_ir: RegisteredFamilySearchIRV62,
    candidate_set: CandidateSetV50,
    model_spec: ModelSpecV50,
    adapter_id: AdapterIdV62,
) -> ExecutableCandidateResolutionV62:
    """Bind an accepted S1 search IR to the current unopened S2 attempt."""

    execution_ir.assert_sealed()
    _sealed_model_spec(model_spec)
    s1_gate_hash = workspace.current_gate("S1")
    if not isinstance(s1_gate_hash, str):
        raise ExecutableCandidateError(
            "S2 resolution requires a current S1 gate"
        )
    if workspace.current_gate("S2") is not None:
        raise ExecutableCandidateError(
            "S2 resolution must be frozen before the S2 gate opens"
        )
    if adapter_id not in execution_ir.allowed_adapter_ids:
        raise ExecutableCandidateError(
            "selected adapter is absent from the S1 execution IR"
        )
    selected = [
        item
        for item in candidate_set.candidates
        if item.candidate_id == model_spec.selected_candidate_id
    ]
    if len(selected) != 1:
        raise ExecutableCandidateError(
            "selected model has no unique S1 candidate"
        )
    candidate = selected[0]
    structural_hash = candidate.structural_hash()
    if (
        structural_hash != model_spec.selected_candidate_structural_hash
        or structural_hash != execution_ir.candidate_structural_hash
        or candidate.candidate_id != execution_ir.candidate_id
    ):
        raise ExecutableCandidateError(
            "S1 candidate, model spec, and execution IR differ"
        )
    if (
        model_spec.assumption_ids != candidate.assumption_ids
        or model_spec.symbol_ids != candidate.symbol_ids
        or model_spec.data_requirement_ids != candidate.data_requirement_ids
    ):
        raise ExecutableCandidateError(
            "selected model semantics differ from the S1 candidate"
        )
    return ExecutableCandidateResolutionV62.seal(
        workspace_spec_hash=_workspace_spec_hash(workspace),
        s1_gate_hash=s1_gate_hash,
        s2_attempt=_workspace_s2_attempt(workspace),
        execution_ir_hash=cast(str, execution_ir.ir_hash),
        model_spec_hash=cast(str, model_spec.model_hash),
        selected_candidate_id=model_spec.selected_candidate_id,
        selected_candidate_structural_hash=structural_hash,
        adapter_id=adapter_id,
        allowed_families=list(
            registered_families_for_adapter_v62(adapter_id)
        ),
        allowed_family_registry_hash=allowed_family_registry_hash_v62(
            adapter_id
        ),
    )


class _ObservedBundle(StrictModel):
    bundle_schema_version: Literal["5.2", "5.7"]
    task_id: Identifier
    bundle_hash: Sha256
    candidate_registry_hash: Sha256
    candidate_graph_hash: Sha256 | None
    nested_candidate_graph_hash: Sha256 | None
    evaluated_families: list[RegisteredFamilyV62]
    evaluated_model_ids: list[Identifier]
    selected_family: RegisteredFamilyV62
    selected_model_id: Identifier
    scientific_acceptance: bool
    fixture_only: bool


def _expected_ode_registry_hash() -> str:
    return sha256_value(
        [
            {
                "candidate_id": family,
                "equation": _ODE_EQUATIONS_V52[family],
            }
            for family in SCALAR_ODE_FAMILIES_V62
        ]
    )


def _expected_hybrid_registry_hash() -> str:
    return sha256_value(
        {
            "families": list(SCALAR_ODE_FAMILIES_V62),
            "residual_modes": list(_HYBRID_RESIDUAL_MODES_V56),
            "graph": _HYBRID_GRAPH_POLICY_V56,
        }
    )


def _observe_ode_bundle(
    bundle: ODEScientificBundleV52,
) -> _ObservedBundle:
    if not bundle.bundle_hash or bundle.bundle_hash != bundle.content_hash():
        raise ExecutableCandidateError("scalar ODE bundle is not sealed")
    candidate_ids = [item.candidate_id for item in bundle.candidates]
    if candidate_ids != list(SCALAR_ODE_FAMILIES_V62):
        raise ExecutableCandidateError(
            "scalar ODE bundle differs from the registered family set"
        )
    if bundle.candidate_registry_hash != _expected_ode_registry_hash():
        raise ExecutableCandidateError(
            "scalar ODE candidate registry hash differs"
        )
    selected = next(
        (
            item
            for item in bundle.candidates
            if item.candidate_id == bundle.selected_candidate_id
        ),
        None,
    )
    if selected is None or bundle.selected_fit_hash != selected.fit.fit_hash:
        raise ExecutableCandidateError(
            "scalar ODE selected family and fit binding differs"
        )
    return _ObservedBundle(
        bundle_schema_version="5.2",
        task_id=bundle.task_id,
        bundle_hash=bundle.bundle_hash,
        candidate_registry_hash=bundle.candidate_registry_hash,
        candidate_graph_hash=None,
        nested_candidate_graph_hash=None,
        evaluated_families=sorted(candidate_ids),
        evaluated_model_ids=sorted(candidate_ids),
        selected_family=bundle.selected_candidate_id,
        selected_model_id=bundle.selected_candidate_id,
        scientific_acceptance=bundle.scientific_acceptance,
        fixture_only=bundle.fixture_only,
    )


def _observe_adaptive_bundle(
    bundle: AdaptivePositiveSeriesBundleV57,
) -> _ObservedBundle:
    if not bundle.bundle_hash or bundle.bundle_hash != bundle.content_hash():
        raise ExecutableCandidateError("adaptive bundle is not sealed")
    primary = bundle.primary_bundle
    graph = bundle.graph
    if not primary.bundle_hash or primary.bundle_hash != primary.content_hash():
        raise ExecutableCandidateError(
            "adaptive primary bundle is not sealed"
        )
    if not graph.graph_hash or graph.graph_hash != graph.content_hash():
        raise ExecutableCandidateError("adaptive graph is not sealed")
    if (
        not primary.graph.graph_hash
        or primary.graph.graph_hash != primary.graph.content_hash()
    ):
        raise ExecutableCandidateError(
            "adaptive nested candidate graph is not sealed"
        )
    if primary.candidate_registry_hash != _expected_hybrid_registry_hash():
        raise ExecutableCandidateError(
            "adaptive primary candidate registry hash differs"
        )
    primary_families = sorted({item.family for item in primary.candidates})
    if primary_families != sorted(SCALAR_ODE_FAMILIES_V62):
        raise ExecutableCandidateError(
            "adaptive primary bundle differs from registered ODE families"
        )
    growth_ids = sorted(item.candidate_id for item in bundle.growth_candidates)
    expected_growth_ids = (
        sorted(
            (
                "log_random_walk_drift",
                "log_growth_ar1",
            )
        )
        if graph.recovery_triggered
        else []
    )
    if growth_ids != expected_growth_ids:
        raise ExecutableCandidateError(
            "adaptive recovery bundle differs from its registered graph"
        )
    primary_statuses = {
        item.level: item.status
        for item in primary.levels
        if item.level != "L0"
    }
    admissible_growth = sorted(
        item.candidate_id
        for item in bundle.growth_candidates
        if item.scientifically_admissible
    )
    if (
        graph.primary_bundle_hash != primary.bundle_hash
        or graph.primary_selected_candidate_id
        != primary.selected_candidate_id
        or graph.primary_level_statuses != primary_statuses
        or graph.recovery_candidate_ids != growth_ids
        or graph.admissible_recovery_candidate_ids != admissible_growth
        or bundle.snapshot_hash != primary.snapshot_hash
        or bundle.task_id != primary.task_id
        or bundle.fixture_only != primary.fixture_only
    ):
        raise ExecutableCandidateError(
            "adaptive outer graph differs from its primary or recovery bundle"
        )
    primary_by_id = {
        item.candidate_id: item for item in primary.candidates
    }
    growth_by_id = {
        item.candidate_id: item for item in bundle.growth_candidates
    }
    if graph.selected_branch == "hybrid_ode":
        selected = primary_by_id.get(graph.selected_model_id)
        if (
            selected is None
            or graph.selected_model_id != primary.selected_candidate_id
        ):
            raise ExecutableCandidateError(
                "adaptive graph selected a different primary model"
            )
        selected_family = cast(RegisteredFamilyV62, selected.family)
    else:
        selected_growth = growth_by_id.get(graph.selected_model_id)
        if selected_growth is None:
            raise ExecutableCandidateError(
                "adaptive graph selected an absent recovery model"
            )
        selected_family = cast(
            RegisteredFamilyV62, selected_growth.candidate_id
        )
    evaluated_model_ids = sorted([*primary_by_id, *growth_by_id])
    evaluated_families = sorted(
        {
            *cast(set[RegisteredFamilyV62], set(primary_families)),
            *cast(set[RegisteredFamilyV62], set(growth_ids)),
        }
    )
    if not set(evaluated_families).issubset(
        set(ADAPTIVE_POSITIVE_SERIES_FAMILIES_V62)
    ):
        raise ExecutableCandidateError(
            "adaptive bundle evaluated an unregistered family"
        )
    return _ObservedBundle(
        bundle_schema_version="5.7",
        task_id=bundle.task_id,
        bundle_hash=bundle.bundle_hash,
        candidate_registry_hash=primary.candidate_registry_hash,
        candidate_graph_hash=graph.graph_hash,
        nested_candidate_graph_hash=primary.graph.graph_hash,
        evaluated_families=evaluated_families,
        evaluated_model_ids=evaluated_model_ids,
        selected_family=selected_family,
        selected_model_id=graph.selected_model_id,
        scientific_acceptance=bundle.scientific_acceptance,
        fixture_only=bundle.fixture_only,
    )


def _observe_bundle(
    *,
    adapter_id: AdapterIdV62,
    bundle: ScientificBundleV62,
) -> _ObservedBundle:
    if adapter_id == SCALAR_ODE_ADAPTER_ID:
        if not isinstance(bundle, ODEScientificBundleV52):
            raise ExecutableCandidateError(
                "scalar ODE resolution received another bundle type"
            )
        return _observe_ode_bundle(bundle)
    if not isinstance(bundle, AdaptivePositiveSeriesBundleV57):
        raise ExecutableCandidateError(
            "adaptive resolution received another bundle type"
        )
    return _observe_adaptive_bundle(bundle)


def build_executable_candidate_receipt_v62(
    *,
    workspace: Any,
    resolution: ExecutableCandidateResolutionV62,
    bundle: ScientificBundleV62,
) -> ExecutableCandidateReceiptV62:
    """Validate the current S3 bundle and seal a non-authoritative receipt."""

    _assert_resolution_current(workspace, resolution)
    s2_gate_hash = workspace.current_gate("S2")
    if not isinstance(s2_gate_hash, str):
        raise ExecutableCandidateError(
            "S3 execution receipt requires a current S2 gate"
        )
    observed = _observe_bundle(
        adapter_id=resolution.adapter_id,
        bundle=bundle,
    )
    workspace_id = getattr(getattr(workspace, "spec", None), "workspace_id", None)
    if (
        isinstance(workspace_id, str)
        and observed.task_id != workspace_id
    ):
        raise ExecutableCandidateError(
            "scientific bundle belongs to another task workspace"
        )
    return ExecutableCandidateReceiptV62.seal(
        workspace_spec_hash=resolution.workspace_spec_hash,
        s1_gate_hash=resolution.s1_gate_hash,
        s2_gate_hash=s2_gate_hash,
        s2_attempt=resolution.s2_attempt,
        resolution_hash=cast(str, resolution.resolution_hash),
        selected_candidate_structural_hash=(
            resolution.selected_candidate_structural_hash
        ),
        adapter_id=resolution.adapter_id,
        allowed_families=resolution.allowed_families,
        allowed_family_registry_hash=(
            resolution.allowed_family_registry_hash
        ),
        evaluated_families=observed.evaluated_families,
        evaluated_model_ids=observed.evaluated_model_ids,
        selected_family=observed.selected_family,
        selected_model_id=observed.selected_model_id,
        bundle_schema_version=observed.bundle_schema_version,
        bundle_task_id=observed.task_id,
        bundle_hash=observed.bundle_hash,
        candidate_registry_hash=observed.candidate_registry_hash,
        candidate_graph_hash=observed.candidate_graph_hash,
        nested_candidate_graph_hash=(
            observed.nested_candidate_graph_hash
        ),
        bundle_scientific_acceptance=observed.scientific_acceptance,
        fixture_only=observed.fixture_only,
    )


def verify_executable_candidate_receipt_v62(
    *,
    workspace: Any,
    resolution: ExecutableCandidateResolutionV62,
    bundle: ScientificBundleV62,
    receipt: ExecutableCandidateReceiptV62,
) -> bool:
    """Replay the receipt from current state and exact bundle semantics."""

    try:
        receipt.assert_sealed()
        expected = build_executable_candidate_receipt_v62(
            workspace=workspace,
            resolution=resolution,
            bundle=bundle,
        )
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        ExecutableCandidateError,
    ):
        return False
    return receipt == expected


__all__ = [
    "ADAPTIVE_POSITIVE_SERIES_ADAPTER_ID",
    "ADAPTIVE_POSITIVE_SERIES_FAMILIES_V62",
    "AdapterIdV62",
    "EXECUTABLE_CANDIDATE_INTENT_PATH",
    "EXECUTABLE_CANDIDATE_IR_PATH",
    "EXECUTABLE_CANDIDATE_RECEIPT_PATH",
    "EXECUTABLE_CANDIDATE_RESOLUTION_PATH",
    "ExecutableCandidateError",
    "ExecutableCandidateReceiptV62",
    "ExecutableCandidateResolutionV62",
    "RegisteredFamilySearchIRV62",
    "RegisteredFamilySearchIntentV62",
    "RegisteredFamilyV62",
    "SCALAR_ODE_ADAPTER_ID",
    "SCALAR_ODE_FAMILIES_V62",
    "SUPPORTED_ADAPTER_IDS_V62",
    "allowed_family_registry_hash_v62",
    "build_executable_candidate_receipt_v62",
    "compile_registered_family_search_ir_v62",
    "registered_families_for_adapter_v62",
    "resolve_executable_candidate_v62",
    "verify_executable_candidate_receipt_v62",
]
