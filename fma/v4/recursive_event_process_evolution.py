from __future__ import annotations

import math
from typing import Annotated, Literal

import numpy as np
from pydantic import Field, model_validator
from scipy.optimize import minimize
from scipy.stats import kstest

from fma.hashing import sha256_value
from fma.schemas import StrictModel
from fma.v2.schemas import Identifier, Sha256

from .codex_open_evolution import (
    GeneratedModelDraftV42,
    OpenEvolutionCandidateViewV42,
    OpenEvolutionFailureViewV42,
    OpenEvolutionGenerationRequestV42,
    OpenEvolutionGenerationTransportV42,
    generation_call_evidence_v42,
)
from .event_process_evolution import (
    EventProcessEvolutionPolicyV41,
    default_event_process_evolution_policy_v41,
    split_event_process_development_v41,
)
from .event_process_open_evolution import (
    EVENT_PROCESS_OPEN_DEVELOPMENT_GATES_V42,
    _renewal_draft_v42,
    _single_hawkes_candidate_v42,
)
from .open_evolution_kernel import (
    DevelopmentEvaluationV42,
    DevelopmentExecutionV42,
    ExecutionAttemptV42,
    HybridEvolutionOperatorV42,
    ModelSymbolV42,
    OpenEvolutionCampaignSpecV42,
    OpenEvolutionOutcomeV42,
    OpenEvolutionProposalV42,
    OpenFailureSignatureV42,
    OpenModelCandidateV42,
    OpenModelGrammarV42,
    PrimitiveApplicationV42,
    PrimitiveRuleV42,
    run_open_evolution_campaign_v42,
)
from .unseen_event_process import (
    EventProcessCandidateV40,
    FittedEventProcessV40,
    USGSCatalogSnapshotV40,
    _confirmation_components,
    _duration_days,
    _poisson_log_likelihood,
    _relative_times,
    fit_event_process_v40,
)


RecursiveExecutorV43 = Literal[
    "homogeneous_poisson",
    "weibull_renewal",
    "exponential_mixture_hawkes",
]
TopologyRuleKindV43 = Literal["exact", "exponential_mixture"]


EVENT_PROCESS_RECURSIVE_ADAPTER_ID_V43 = "event_process_recursive_v43"


def _hash_without(model: StrictModel, field: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude={field}))


class TopologyCompilerRuleV43(StrictModel):
    schema_version: Literal["4.3"] = "4.3"
    rule_id: Identifier
    rule_kind: TopologyRuleKindV43
    executor_key: RecursiveExecutorV43
    exact_primitive_counts: dict[Identifier, Annotated[int, Field(ge=0, le=16)]]
    minimum_components: Annotated[int, Field(ge=0, le=16)] = 0
    maximum_components: Annotated[int, Field(ge=0, le=16)] = 0

    @model_validator(mode="after")
    def validate_rule(self) -> "TopologyCompilerRuleV43":
        if self.rule_kind == "exact":
            if self.minimum_components or self.maximum_components:
                raise ValueError("exact topology rule cannot define components")
            if not self.exact_primitive_counts:
                raise ValueError("exact topology rule needs primitive counts")
        else:
            if self.executor_key != "exponential_mixture_hawkes":
                raise ValueError("mixture rule needs the mixture executor")
            if not 1 <= self.minimum_components <= self.maximum_components:
                raise ValueError("mixture component range is invalid")
            if self.exact_primitive_counts != {"constant_background": 1}:
                raise ValueError("mixture rule needs exactly one background source")
        return self


class TopologyCompilerRegistryV43(StrictModel):
    schema_version: Literal["4.3"] = "4.3"
    registry_id: Identifier
    rules: Annotated[list[TopologyCompilerRuleV43], Field(min_length=1, max_length=32)]
    registry_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_registry(self) -> "TopologyCompilerRegistryV43":
        rule_ids = [item.rule_id for item in self.rules]
        if rule_ids != sorted(set(rule_ids)):
            raise ValueError("topology rules must be sorted and unique")
        executor_keys = [item.executor_key for item in self.rules]
        if len(executor_keys) != len(set(executor_keys)):
            raise ValueError("registry executor keys must be unique")
        if self.registry_hash and self.registry_hash != self.content_hash():
            raise ValueError("registry_hash does not match topology registry")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "registry_hash")

    def assert_sealed(self) -> None:
        if not self.registry_hash or self.registry_hash != self.content_hash():
            raise ValueError("topology compiler registry is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "TopologyCompilerRegistryV43":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"registry_hash"}),
            registry_hash=draft.content_hash(),
        )


def event_process_topology_registry_v43() -> TopologyCompilerRegistryV43:
    return TopologyCompilerRegistryV43.seal(
        registry_id="event_process_topology_registry_v43",
        rules=[
            TopologyCompilerRuleV43(
                rule_id="exponential_mixture_hawkes",
                rule_kind="exponential_mixture",
                executor_key="exponential_mixture_hawkes",
                exact_primitive_counts={"constant_background": 1},
                minimum_components=1,
                maximum_components=4,
            ),
            TopologyCompilerRuleV43(
                rule_id="homogeneous_poisson",
                rule_kind="exact",
                executor_key="homogeneous_poisson",
                exact_primitive_counts={"constant_background": 1},
            ),
            TopologyCompilerRuleV43(
                rule_id="weibull_renewal",
                rule_kind="exact",
                executor_key="weibull_renewal",
                exact_primitive_counts={"renewal_hazard": 1},
            ),
        ],
    )


EVENT_PROCESS_RECURSIVE_REGISTRY_HASH_V43 = (
    event_process_topology_registry_v43().registry_hash
)
EVENT_PROCESS_RECURSIVE_ADAPTER_HASH_V43 = sha256_value(
    {
        "adapter": EVENT_PROCESS_RECURSIVE_ADAPTER_ID_V43,
        "registry_hash": EVENT_PROCESS_RECURSIVE_REGISTRY_HASH_V43,
        "compiler": "declarative_rule_match_connected_add_tree_v43",
        "mixture_fit": "ordered_k_component_exponential_hawkes_mle_v43",
        "component_range": [1, 4],
        "evaluation": "development_70_30_nine_frozen_gates_v43",
        "private_data_access": False,
    }
)


class CompiledRecursiveEventProcessV43(StrictModel):
    schema_version: Literal["4.3"] = "4.3"
    candidate_hash: Sha256
    registry_hash: Sha256
    rule_id: Identifier
    executor_key: RecursiveExecutorV43
    component_count: Annotated[int, Field(ge=0, le=16)]
    primitive_counts: dict[Identifier, Annotated[int, Field(ge=1, le=16)]]
    structural_signature: Sha256
    application_ids: Annotated[list[Identifier], Field(min_length=1, max_length=32)]
    compiled_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_compiled(self) -> "CompiledRecursiveEventProcessV43":
        if self.application_ids != sorted(set(self.application_ids)):
            raise ValueError("compiled application IDs must be sorted and unique")
        if self.compiled_hash and self.compiled_hash != self.content_hash():
            raise ValueError("compiled_hash does not match recursive event process")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "compiled_hash")

    def assert_sealed(self) -> None:
        if not self.compiled_hash or self.compiled_hash != self.content_hash():
            raise ValueError("compiled recursive event process is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "CompiledRecursiveEventProcessV43":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"compiled_hash"}),
            compiled_hash=draft.content_hash(),
        )


class FittedRecursiveEventProcessV43(StrictModel):
    schema_version: Literal["4.3"] = "4.3"
    candidate_hash: Sha256
    compiled_hash: Sha256
    development_snapshot_hash: Sha256
    executor_key: RecursiveExecutorV43
    component_count: Annotated[int, Field(ge=0, le=16)]
    parameters: dict[
        Identifier, Annotated[float, Field(allow_inf_nan=False)]
    ]
    optimizer_converged: bool
    optimizer_message: Annotated[str, Field(min_length=1, max_length=2000)]
    development_log_likelihood: Annotated[
        float, Field(allow_inf_nan=False)
    ]
    development_bic: Annotated[float, Field(allow_inf_nan=False)]
    independent_parameter_count: Annotated[int, Field(ge=1, le=64)]
    fit_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_fit(self) -> "FittedRecursiveEventProcessV43":
        if self.executor_key == "exponential_mixture_hawkes":
            expected = 2 * self.component_count + 1
            if self.component_count < 1 or self.independent_parameter_count != expected:
                raise ValueError("mixture fit parameter count is inconsistent")
        elif self.component_count != 0:
            raise ValueError("non-mixture fit cannot contain memory components")
        if self.fit_hash and self.fit_hash != self.content_hash():
            raise ValueError("fit_hash does not match recursive event-process fit")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "fit_hash")

    def assert_sealed(self) -> None:
        if not self.fit_hash or self.fit_hash != self.content_hash():
            raise ValueError("recursive event-process fit is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "FittedRecursiveEventProcessV43":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"fit_hash"}),
            fit_hash=draft.content_hash(),
        )


def event_process_recursive_grammar_v43() -> OpenModelGrammarV42:
    return OpenModelGrammarV42.seal(
        grammar_id="event_process_recursive_grammar_v43",
        seed_families=["single_exponential_hawkes"],
        primitives=[
            PrimitiveRuleV42(
                primitive_id="add_rate",
                role="combiner",
                input_units=["rate_per_day", "rate_per_day"],
                output_unit="rate_per_day",
                max_uses_per_candidate=4,
            ),
            PrimitiveRuleV42(
                primitive_id="constant_background",
                role="source",
                input_units=["rate_per_day"],
                output_unit="rate_per_day",
            ),
            PrimitiveRuleV42(
                primitive_id="exponential_memory",
                role="mechanism",
                input_units=["event_history", "unitless", "rate_per_day"],
                output_unit="rate_per_day",
                max_uses_per_candidate=4,
            ),
            PrimitiveRuleV42(
                primitive_id="renewal_hazard",
                role="mechanism",
                input_units=["day", "unitless", "day"],
                output_unit="rate_per_day",
            ),
        ],
        executable_adapter_ids=[EVENT_PROCESS_RECURSIVE_ADAPTER_ID_V43],
        executable_adapter_hashes={
            EVENT_PROCESS_RECURSIVE_ADAPTER_ID_V43: (
                EVENT_PROCESS_RECURSIVE_ADAPTER_HASH_V43
            )
        },
        forbidden_tokens=["private_confirmation", "qualification_granted"],
        max_symbols=28,
        max_applications=10,
    )


def _reachable_application_ids(
    candidate: OpenModelCandidateV42,
) -> tuple[set[str], dict[str, PrimitiveApplicationV42]]:
    symbols = {item.symbol_id: item for item in candidate.symbols}
    intensity = symbols.get("intensity")
    if intensity is None or intensity.role != "output" or intensity.unit != "rate_per_day":
        raise ValueError("recursive event process needs intensity: rate_per_day")
    producers: dict[str, PrimitiveApplicationV42] = {}
    consumers: dict[str, int] = {}
    for application in candidate.applications:
        if application.output in producers:
            raise ValueError("recursive event process has duplicate producers")
        producers[application.output] = application
        for symbol_id in application.inputs:
            consumers[symbol_id] = consumers.get(symbol_id, 0) + 1
    if "intensity" not in producers:
        raise ValueError("recursive event process does not produce intensity")
    reachable: set[str] = set()
    visited_symbols: set[str] = set()

    def visit(symbol_id: str) -> None:
        if symbol_id in visited_symbols:
            return
        visited_symbols.add(symbol_id)
        producer = producers.get(symbol_id)
        if producer is None:
            return
        reachable.add(producer.application_id)
        for input_symbol in producer.inputs:
            visit(input_symbol)

    visit("intensity")
    all_ids = {item.application_id for item in candidate.applications}
    if reachable != all_ids:
        raise ValueError("recursive event process contains unused applications")
    for output_symbol in producers:
        if output_symbol == "intensity":
            continue
        if consumers.get(output_symbol) != 1:
            raise ValueError("recursive expression is not a single connected tree")
    return reachable, producers


def _assert_mixture_semantics(
    candidate: OpenModelCandidateV42,
    component_count: int,
) -> None:
    symbols = {item.symbol_id: item for item in candidate.symbols}
    memories = [
        item
        for item in candidate.applications
        if item.primitive_id == "exponential_memory"
    ]
    if len(memories) != component_count:
        raise ValueError("mixture component count changed during compilation")
    histories = {item.inputs[0] for item in memories}
    branchings = [item.inputs[1] for item in memories]
    decays = [item.inputs[2] for item in memories]
    if len(histories) != 1:
        raise ValueError("mixture components must share one observed event history")
    history = symbols[next(iter(histories))]
    if history.role != "observed" or history.unit != "event_history":
        raise ValueError("mixture history must be an observed event history")
    if len(set(branchings)) != component_count or len(set(decays)) != component_count:
        raise ValueError("mixture components need distinct parameter symbols")
    for symbol_id in branchings + decays:
        if symbols[symbol_id].role != "parameter":
            raise ValueError("mixture branching and decay inputs must be parameters")
    backgrounds = [
        item
        for item in candidate.applications
        if item.primitive_id == "constant_background"
    ]
    if len(backgrounds) != 1:
        raise ValueError("mixture needs exactly one background source")
    if symbols[backgrounds[0].inputs[0]].role != "parameter":
        raise ValueError("background input must be a parameter")


def compile_recursive_event_process_v43(
    candidate: OpenModelCandidateV42,
    registry: TopologyCompilerRegistryV43 | None = None,
) -> CompiledRecursiveEventProcessV43:
    """Match a candidate against frozen declarative rules, never its family label."""

    candidate.assert_sealed()
    frozen_registry = registry or event_process_topology_registry_v43()
    frozen_registry.assert_sealed()
    reachable, _ = _reachable_application_ids(candidate)
    counts: dict[str, int] = {}
    for application in candidate.applications:
        counts[application.primitive_id] = (
            counts.get(application.primitive_id, 0) + 1
        )
    matches: list[tuple[TopologyCompilerRuleV43, int]] = []
    for rule in frozen_registry.rules:
        if rule.rule_kind == "exact":
            if counts == rule.exact_primitive_counts:
                matches.append((rule, 0))
            continue
        component_count = counts.get("exponential_memory", 0)
        if (
            rule.minimum_components
            <= component_count
            <= rule.maximum_components
            and counts
            == {
                "add_rate": component_count,
                "constant_background": 1,
                "exponential_memory": component_count,
            }
        ):
            _assert_mixture_semantics(candidate, component_count)
            matches.append((rule, component_count))
    if len(matches) != 1:
        raise ValueError(
            f"recursive topology must match exactly one rule; matched {len(matches)}"
        )
    rule, component_count = matches[0]
    signature = sha256_value(
        {
            "registry_hash": frozen_registry.registry_hash,
            "rule_id": rule.rule_id,
            "executor_key": rule.executor_key,
            "component_count": component_count,
            "primitive_counts": counts,
        }
    )
    return CompiledRecursiveEventProcessV43.seal(
        candidate_hash=candidate.candidate_hash,
        registry_hash=frozen_registry.registry_hash,
        rule_id=rule.rule_id,
        executor_key=rule.executor_key,
        component_count=component_count,
        primitive_counts=counts,
        structural_signature=signature,
        application_ids=sorted(reachable),
    )


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - float(np.max(values))
    weights = np.exp(shifted)
    return weights / float(np.sum(weights))


def _mixture_parameters(
    values: np.ndarray,
    component_count: int,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    background = float(values[0])
    total_branching = float(values[1])
    if component_count == 1:
        weights = np.asarray([1.0], dtype=float)
        decay_start = 2
    else:
        logits = np.concatenate(
            (values[2 : 2 + component_count - 1], np.asarray([0.0]))
        )
        weights = _softmax(logits)
        decay_start = 2 + component_count - 1
    decays = np.sort(
        np.exp(values[decay_start : decay_start + component_count])
    )[::-1]
    return background, total_branching, weights, decays


def _exponential_mixture_components(
    events: np.ndarray,
    *,
    start: float,
    end: float,
    history: np.ndarray,
    background: float,
    total_branching: float,
    weights: np.ndarray,
    decays: np.ndarray,
) -> tuple[float, list[float], float]:
    if (
        background <= 0
        or not 0 <= total_branching < 1
        or len(weights) == 0
        or len(weights) != len(decays)
        or np.any(weights <= 0)
        or not math.isclose(float(np.sum(weights)), 1.0, rel_tol=1e-8)
        or np.any(decays <= 0)
    ):
        return -math.inf, [], math.inf
    branchings = total_branching * weights
    prior = [float(value) for value in history if value < start]
    log_terms: list[float] = []
    rescaled: list[float] = []
    left = start
    compensator = 0.0

    def memory_intensity(event_time: float) -> float:
        return sum(
            float(branching)
            * float(decay)
            * math.exp(-float(decay) * (event_time - past))
            for branching, decay in zip(branchings, decays, strict=True)
            for past in prior
            if past < event_time
        )

    def memory_integral(interval_start: float, interval_end: float) -> float:
        return sum(
            float(branching)
            * (
                math.exp(
                    -float(decay) * max(interval_start - past, 0.0)
                )
                - math.exp(-float(decay) * (interval_end - past))
            )
            for branching, decay in zip(branchings, decays, strict=True)
            for past in prior
            if past <= interval_start
        )

    for event in events:
        event_time = float(event)
        intensity = background + memory_intensity(event_time)
        if not math.isfinite(intensity) or intensity <= 0:
            return -math.inf, [], math.inf
        increment = (
            background * (event_time - left)
            + memory_integral(left, event_time)
        )
        if not math.isfinite(increment) or increment < 0:
            return -math.inf, [], math.inf
        log_terms.append(math.log(intensity))
        rescaled.append(increment)
        compensator += increment
        prior.append(event_time)
        left = event_time
    compensator += (
        background * (end - left) + memory_integral(left, end)
    )
    return sum(log_terms) - compensator, rescaled, compensator


def fit_recursive_event_process_v43(
    candidate: OpenModelCandidateV42,
    development: USGSCatalogSnapshotV40,
    registry: TopologyCompilerRegistryV43 | None = None,
) -> tuple[CompiledRecursiveEventProcessV43, FittedRecursiveEventProcessV43]:
    candidate.assert_sealed()
    development.assert_sealed()
    compiled = compile_recursive_event_process_v43(candidate, registry)
    if compiled.executor_key != "exponential_mixture_hawkes":
        domain_family = {
            "homogeneous_poisson": "homogeneous_poisson",
            "weibull_renewal": "weibull_renewal",
        }[compiled.executor_key]
        domain_candidate = EventProcessCandidateV40.seal(
            family=domain_family,
            hawkes_branching_initial=0.45,
            hawkes_decay_days_initial=2.0,
            rationale=(
                "Compile a registry-admitted exact event-process rule into "
                "the existing maximum-likelihood implementation."
            ),
            expected_failure_modes=[
                "The exact compiled structure may fail frozen development gates."
            ],
        )
        domain_fit = fit_event_process_v40(domain_candidate, development)
        return compiled, FittedRecursiveEventProcessV43.seal(
            candidate_hash=candidate.candidate_hash,
            compiled_hash=compiled.compiled_hash,
            development_snapshot_hash=development.snapshot_hash,
            executor_key=compiled.executor_key,
            component_count=0,
            parameters=domain_fit.parameters,
            optimizer_converged=domain_fit.optimizer_converged,
            optimizer_message=domain_fit.optimizer_message,
            development_log_likelihood=domain_fit.development_log_likelihood,
            development_bic=domain_fit.development_bic,
            independent_parameter_count=domain_fit.parameter_count,
        )

    events = _relative_times(development)
    duration = _duration_days(development.query)
    event_count = len(events)
    rate = event_count / duration
    component_count = compiled.component_count
    upper_background = max(10.0, 10.0 * rate)
    decay_sets = [
        np.geomspace(2.0, 1.0 / 14.0, component_count),
        np.geomspace(10.0, 1.0 / 30.0, component_count),
        np.geomspace(0.5, 1.0 / 60.0, component_count),
    ]
    starts: list[np.ndarray] = []
    for index, decays in enumerate(decay_sets):
        logits = np.zeros(max(component_count - 1, 0), dtype=float)
        starts.append(
            np.concatenate(
                (
                    np.asarray(
                        [
                            max(rate * [0.55, 0.75, 0.35][index], 1e-4),
                            [0.45, 0.25, 0.65][index],
                        ]
                    ),
                    logits,
                    np.log(decays),
                )
            )
        )

    def objective(values: np.ndarray) -> float:
        background, branching, weights, decays = _mixture_parameters(
            values, component_count
        )
        likelihood, _, _ = _exponential_mixture_components(
            events,
            start=0.0,
            end=duration,
            history=np.asarray([], dtype=float),
            background=background,
            total_branching=branching,
            weights=weights,
            decays=decays,
        )
        return -likelihood if math.isfinite(likelihood) else 1e100

    bounds = (
        [(1e-6, upper_background), (1e-6, 0.949999)]
        + [(-5.0, 5.0)] * max(component_count - 1, 0)
        + [(math.log(1.0 / 60.0), math.log(10.0))] * component_count
    )
    results = [
        minimize(
            objective,
            start_values,
            method="L-BFGS-B",
            bounds=bounds,
        )
        for start_values in starts
    ]
    result = min(results, key=lambda item: float(item.fun))
    background, total_branching, weights, decays = _mixture_parameters(
        result.x, component_count
    )
    parameters: dict[str, float] = {
        "background_rate_per_day": background,
        "total_branching_ratio": total_branching,
    }
    for index, (weight, decay) in enumerate(
        zip(weights, decays, strict=True)
    ):
        parameters[f"component_{index}_weight"] = float(weight)
        parameters[f"component_{index}_branching_ratio"] = float(
            total_branching * weight
        )
        parameters[f"component_{index}_decay_rate_per_day"] = float(decay)
    log_likelihood = -float(result.fun)
    independent_parameter_count = 2 * component_count + 1
    bic = (
        independent_parameter_count * math.log(event_count)
        - 2.0 * log_likelihood
    )
    return compiled, FittedRecursiveEventProcessV43.seal(
        candidate_hash=candidate.candidate_hash,
        compiled_hash=compiled.compiled_hash,
        development_snapshot_hash=development.snapshot_hash,
        executor_key=compiled.executor_key,
        component_count=component_count,
        parameters=parameters,
        optimizer_converged=bool(
            result.success and math.isfinite(log_likelihood)
        ),
        optimizer_message=str(result.message),
        development_log_likelihood=log_likelihood,
        development_bic=bic,
        independent_parameter_count=independent_parameter_count,
    )


def _validation_components_v43(
    fit: FittedRecursiveEventProcessV43,
    training: USGSCatalogSnapshotV40,
    validation: USGSCatalogSnapshotV40,
) -> tuple[float, list[float], float]:
    if fit.executor_key != "exponential_mixture_hawkes":
        family = {
            "homogeneous_poisson": "homogeneous_poisson",
            "weibull_renewal": "weibull_renewal",
        }[fit.executor_key]
        domain_fit = FittedEventProcessV40.seal(
            candidate_hash=fit.candidate_hash,
            development_snapshot_hash=training.snapshot_hash,
            family=family,
            parameters=fit.parameters,
            optimizer_converged=fit.optimizer_converged,
            optimizer_message=fit.optimizer_message,
            development_log_likelihood=fit.development_log_likelihood,
            development_bic=fit.development_bic,
            parameter_count=fit.independent_parameter_count,
        )
        return _confirmation_components(domain_fit, training, validation)
    weights = np.asarray(
        [
            fit.parameters[f"component_{index}_weight"]
            for index in range(fit.component_count)
        ]
    )
    decays = np.asarray(
        [
            fit.parameters[f"component_{index}_decay_rate_per_day"]
            for index in range(fit.component_count)
        ]
    )
    history = np.asarray(
        [
            (event.origin_time - validation.query.start).total_seconds()
            / 86400.0
            for event in training.events
        ],
        dtype=float,
    )
    return _exponential_mixture_components(
        _relative_times(validation),
        start=0.0,
        end=_duration_days(validation.query),
        history=history,
        background=fit.parameters["background_rate_per_day"],
        total_branching=fit.parameters["total_branching_ratio"],
        weights=weights,
        decays=decays,
    )


def _recursive_seed_candidate_v43() -> OpenModelCandidateV42:
    seed = _single_hawkes_candidate_v42()
    return OpenModelCandidateV42.seal(
        **{
            **seed.model_dump(
                mode="json",
                exclude={"candidate_hash", "candidate_id", "executable_adapter_id"},
            ),
            "candidate_id": "event.recursive.single_hawkes.g0",
            "executable_adapter_id": EVENT_PROCESS_RECURSIVE_ADAPTER_ID_V43,
        }
    )


def _candidate_from_draft_v43(
    draft: GeneratedModelDraftV42,
    *,
    source: Literal["prescribed", "generated"],
    parent: OpenModelCandidateV42,
    operator: HybridEvolutionOperatorV42,
    next_generation: int,
) -> OpenModelCandidateV42:
    return OpenModelCandidateV42.seal(
        candidate_id=(
            f"event.recursive.{draft.family}.g{next_generation}."
            f"{parent.candidate_hash[:8]}"
        ),
        generation=next_generation,
        family=draft.family,
        source=source,
        proposed_by="harness" if source == "prescribed" else "model",
        executable_adapter_id=EVENT_PROCESS_RECURSIVE_ADAPTER_ID_V43,
        symbols=draft.symbols,
        applications=draft.applications,
        assumptions=draft.assumptions,
        parent_candidate_hashes=[parent.candidate_hash],
        operator_hashes=[operator.operator_hash],
        rationale=draft.rationale,
        expected_failure_modes=draft.expected_failure_modes,
    )


class RecursiveEventProcessEvolutionAdapterV43:
    adapter_id = EVENT_PROCESS_RECURSIVE_ADAPTER_ID_V43
    adapter_contract_hash = EVENT_PROCESS_RECURSIVE_ADAPTER_HASH_V43

    def __init__(
        self,
        development: USGSCatalogSnapshotV40,
        transport: OpenEvolutionGenerationTransportV42,
        policy: EventProcessEvolutionPolicyV41 | None = None,
        registry: TopologyCompilerRegistryV43 | None = None,
    ) -> None:
        development.assert_sealed()
        self.development = development
        self.transport = transport
        self.policy = policy or default_event_process_evolution_policy_v41()
        self.policy.assert_sealed()
        self.registry = registry or event_process_topology_registry_v43()
        self.registry.assert_sealed()
        self.training, self.validation = split_event_process_development_v41(
            development, self.policy.split_fraction
        )

    def _assert_spec(
        self,
        spec: OpenEvolutionCampaignSpecV42,
        grammar: OpenModelGrammarV42 | None = None,
    ) -> None:
        if spec.development_data_hash != self.development.snapshot_hash:
            raise ValueError("recursive campaign refers to another snapshot")
        if spec.required_development_gates != EVENT_PROCESS_OPEN_DEVELOPMENT_GATES_V42:
            raise ValueError("recursive campaign gate set differs from policy")
        if spec.evaluation_policy.get("policy_hash") != self.policy.policy_hash:
            raise ValueError("recursive campaign policy hash differs")
        if (
            spec.evaluation_policy.get("topology_registry_hash")
            != self.registry.registry_hash
        ):
            raise ValueError("recursive campaign registry hash differs")
        if grammar is not None and grammar.grammar_hash != spec.grammar_hash:
            raise ValueError("recursive campaign refers to another grammar")

    def initial_candidates(
        self,
        spec: OpenEvolutionCampaignSpecV42,
        grammar: OpenModelGrammarV42,
    ) -> list[OpenModelCandidateV42]:
        self._assert_spec(spec, grammar)
        return [_recursive_seed_candidate_v43()]

    def supports_candidate(self, candidate: OpenModelCandidateV42) -> bool:
        try:
            compile_recursive_event_process_v43(candidate, self.registry)
            return True
        except (TypeError, ValueError):
            return False

    def execute(
        self,
        spec: OpenEvolutionCampaignSpecV42,
        candidate: OpenModelCandidateV42,
        attempt: ExecutionAttemptV42,
    ) -> DevelopmentExecutionV42:
        self._assert_spec(spec)
        if attempt.candidate_hash != candidate.candidate_hash:
            raise ValueError("recursive attempt belongs to another candidate")
        compiled, fit = fit_recursive_event_process_v43(
            candidate, self.training, self.registry
        )
        return DevelopmentExecutionV42.seal(
            candidate_hash=candidate.candidate_hash,
            attempt_hash=attempt.attempt_hash,
            idempotency_key=attempt.idempotency_key,
            development_data_hash=self.development.snapshot_hash,
            converged=fit.optimizer_converged,
            metrics={
                "training_bic": fit.development_bic,
                "training_log_likelihood": fit.development_log_likelihood,
                "parameter_count": float(fit.independent_parameter_count),
                "component_count": float(fit.component_count),
            },
            domain_payload={
                "compiled": compiled.model_dump(mode="json"),
                "fit": fit.model_dump(mode="json"),
                "training_snapshot_hash": self.training.snapshot_hash,
                "validation_snapshot_hash": self.validation.snapshot_hash,
                "training_event_count": len(self.training.events),
                "validation_event_count": len(self.validation.events),
            },
        )

    def evaluate(
        self,
        spec: OpenEvolutionCampaignSpecV42,
        candidate: OpenModelCandidateV42,
        execution: DevelopmentExecutionV42,
    ) -> DevelopmentEvaluationV42:
        self._assert_spec(spec)
        fit_payload = execution.domain_payload.get("fit")
        if not isinstance(fit_payload, dict):
            raise ValueError("recursive execution lacks a fitted model")
        fit = FittedRecursiveEventProcessV43.model_validate(fit_payload)
        fit.assert_sealed()
        if fit.candidate_hash != candidate.candidate_hash:
            raise ValueError("recursive fit belongs to another candidate")

        training_duration = _duration_days(self.training.query)
        baseline_rate = len(self.training.events) / training_duration
        baseline_training_ll = _poisson_log_likelihood(
            baseline_rate, len(self.training.events), training_duration
        )
        baseline_training_bic = (
            math.log(len(self.training.events)) - 2.0 * baseline_training_ll
        )
        validation_duration = _duration_days(self.validation.query)
        baseline_validation_ll = _poisson_log_likelihood(
            baseline_rate, len(self.validation.events), validation_duration
        )
        candidate_validation_ll, rescaled, compensator = (
            _validation_components_v43(fit, self.training, self.validation)
        )
        validation_count = len(self.validation.events)
        log_score_lift = (
            candidate_validation_ll - baseline_validation_ll
        ) / validation_count
        ks_pvalue = (
            float(kstest(np.asarray(rescaled, dtype=float), "expon").pvalue)
            if rescaled
            and all(math.isfinite(value) and value >= 0 for value in rescaled)
            else 0.0
        )
        count_error = abs(compensator - validation_count) / validation_count

        branching = 0.0
        component_count = fit.component_count
        minimum_component_branching = 0.0
        minimum_decay_ratio = 0.0
        maximum_decay = 0.0
        if fit.executor_key == "exponential_mixture_hawkes":
            branching = fit.parameters["total_branching_ratio"]
            component_branchings = [
                fit.parameters[f"component_{index}_branching_ratio"]
                for index in range(component_count)
            ]
            decays = [
                fit.parameters[f"component_{index}_decay_rate_per_day"]
                for index in range(component_count)
            ]
            weights = [
                fit.parameters[f"component_{index}_weight"]
                for index in range(component_count)
            ]
            minimum_component_branching = min(component_branchings)
            maximum_decay = max(decays)
            if component_count > 1:
                decay_ratios = [
                    decays[index] / decays[index + 1]
                    for index in range(component_count - 1)
                ]
                minimum_decay_ratio = min(decay_ratios)
                component_identifiability = (
                    minimum_component_branching >= 0.02
                    and minimum_decay_ratio >= 2.0
                )
            else:
                component_identifiability = True
            parameter_interior = (
                0.001 < branching < 0.949
                and all(
                    1.0 / 59.99
                    < value
                    < self.policy.maximum_interior_hawkes_decay_rate_per_day
                    for value in decays
                )
                and (
                    component_count == 1
                    or all(0.021 < value < 0.979 for value in weights)
                )
            )
        elif fit.executor_key == "weibull_renewal":
            component_identifiability = True
            parameter_interior = (
                0.1001 < fit.parameters["shape"] < 9.999
                and 0.00011 < fit.parameters["scale_days"] < 99.99
            )
        else:
            component_identifiability = True
            parameter_interior = True

        gates = {
            "compensator_count_calibration": (
                count_error
                <= self.policy.maximum_compensator_count_relative_error
            ),
            "component_identifiability": component_identifiability,
            "development_slice_contract": (
                len(self.training.events) >= self.policy.minimum_events_per_slice
                and len(self.validation.events)
                >= self.policy.minimum_events_per_slice
            ),
            "hawkes_stationarity": (
                fit.executor_key != "exponential_mixture_hawkes"
                or branching < self.policy.maximum_hawkes_branching_ratio
            ),
            "optimizer_converged": fit.optimizer_converged,
            "parameter_interior": parameter_interior,
            "time_rescaling_calibration": (
                ks_pvalue >= self.policy.minimum_time_rescaling_ks_pvalue
            ),
            "training_bic_not_worse": (
                fit.development_bic <= baseline_training_bic
            ),
            "validation_log_score_lift": (
                log_score_lift
                >= self.policy.minimum_validation_log_score_lift_nat_per_event
            ),
        }
        diagnostics_by_gate = {
            "optimizer_converged": "optimizer_failure",
            "training_bic_not_worse": "unsupported_complexity",
            "validation_log_score_lift": "no_predictive_lift",
            "time_rescaling_calibration": "time_rescaling_miscalibration",
            "parameter_interior": "parameter_boundary",
            "hawkes_stationarity": "hawkes_nonstationarity",
            "development_slice_contract": "insufficient_development_slice",
            "component_identifiability": "components_not_identifiable",
        }
        diagnostics = [
            code
            for gate, code in diagnostics_by_gate.items()
            if not gates[gate]
        ]
        if not gates["compensator_count_calibration"]:
            diagnostics.append(
                "count_underprediction"
                if compensator < validation_count
                else "count_overprediction"
            )
        ks_term = 0.05 * math.log10(
            max(ks_pvalue, 1e-15)
            / self.policy.minimum_time_rescaling_ks_pvalue
        )
        failed_gate_penalty = 0.05 * sum(not value for value in gates.values())
        utility = log_score_lift - count_error + ks_term - failed_gate_penalty
        return DevelopmentEvaluationV42.seal(
            candidate_hash=candidate.candidate_hash,
            execution_hash=execution.execution_hash,
            evaluator_epoch=spec.evaluator_epoch,
            gates=gates,
            metrics={
                "baseline_training_bic": baseline_training_bic,
                "candidate_training_bic": fit.development_bic,
                "baseline_validation_log_likelihood": baseline_validation_ll,
                "candidate_validation_log_likelihood": candidate_validation_ll,
                "validation_log_score_lift_nat_per_event": log_score_lift,
                "time_rescaling_ks_pvalue": ks_pvalue,
                "compensator_total": compensator,
                "observed_validation_count": float(validation_count),
                "compensator_count_relative_error": count_error,
                "branching_ratio": branching,
                "component_count": float(component_count),
                "minimum_component_branching_ratio": (
                    minimum_component_branching
                ),
                "minimum_adjacent_decay_ratio": minimum_decay_ratio,
                "maximum_decay_rate_per_day": maximum_decay,
            },
            utility=utility,
            diagnostic_codes=sorted(set(diagnostics)),
            disposition=(
                "advance"
                if all(gates.values())
                else "discard"
                if not fit.optimizer_converged
                else "mutate"
            ),
        )

    def prescribed_evolve(
        self,
        spec: OpenEvolutionCampaignSpecV42,
        grammar: OpenModelGrammarV42,
        candidate: OpenModelCandidateV42,
        evaluation: DevelopmentEvaluationV42,
        failure: OpenFailureSignatureV42,
        next_generation: int,
    ) -> list[OpenEvolutionProposalV42]:
        self._assert_spec(spec, grammar)
        if evaluation.disposition == "discard":
            return []
        draft = _renewal_draft_v42()
        operator = HybridEvolutionOperatorV42.seal(
            operator_id=(
                f"prescribed.recursive.renewal.g{next_generation}."
                f"{candidate.candidate_hash[:8]}"
            ),
            channel="prescribed",
            kind=draft.kind,
            proposed_by="harness",
            source_candidate_hash=candidate.candidate_hash,
            source_evaluation_hash=evaluation.evaluation_hash,
            failure_signature_hash=failure.failure_hash,
            target_family=draft.family,
            transformation_summary=draft.transformation_summary,
            rationale=draft.rationale,
        )
        return [
            OpenEvolutionProposalV42(
                operator=operator,
                candidate=_candidate_from_draft_v43(
                    draft,
                    source="prescribed",
                    parent=candidate,
                    operator=operator,
                    next_generation=next_generation,
                ),
                priority=draft.priority,
            )
        ]

    def generated_evolve(
        self,
        spec: OpenEvolutionCampaignSpecV42,
        grammar: OpenModelGrammarV42,
        candidate: OpenModelCandidateV42,
        evaluation: DevelopmentEvaluationV42,
        failure: OpenFailureSignatureV42,
        next_generation: int,
    ) -> list[OpenEvolutionProposalV42]:
        self._assert_spec(spec, grammar)
        if (
            evaluation.disposition == "discard"
            or spec.generated_quota_per_failure == 0
        ):
            return []
        current = compile_recursive_event_process_v43(
            candidate, self.registry
        )
        if current.executor_key != "exponential_mixture_hawkes":
            return []
        next_component_count = min(current.component_count + 1, 4)
        mutation_guidance = (
            f"The current expression has {current.component_count} "
            "exponential_memory components. For parameter_boundary, add "
            f"exactly one component and return {next_component_count} total; "
            "reuse the same event_history, give every component distinct "
            "branching and decay parameter symbols, and connect one background "
            "plus all memory rates with a binary add_rate tree."
        )
        request = OpenEvolutionGenerationRequestV42.seal(
            request_id=(
                f"recursive.generate.{candidate.candidate_hash[:12]}."
                f"g{next_generation}"
            ),
            objective=spec.objective,
            grammar_id=grammar.grammar_id,
            grammar_hash=grammar.grammar_hash,
            allowed_primitives=grammar.primitives,
            executable_adapter_id=self.adapter_id,
            max_symbols=grammar.max_symbols,
            max_applications=grammar.max_applications,
            current_candidate=OpenEvolutionCandidateViewV42(
                family=candidate.family,
                symbols=candidate.symbols,
                applications=candidate.applications,
                assumptions=candidate.assumptions,
                rationale=candidate.rationale,
            ),
            failure=OpenEvolutionFailureViewV42(
                failed_gates=failure.failed_gates,
                diagnostic_codes=failure.diagnostic_codes,
                sanitized_summary=failure.sanitized_summary,
            ),
            development_metrics=evaluation.metrics,
            adapter_guidance=[
                (
                    "Registry rule: exponential_mixture_hawkes accepts one "
                    "constant_background, K=1..4 exponential_memory applications, "
                    "and exactly K add_rate applications."
                ),
                (
                    "Every application must reach intensity and every produced "
                    "intermediate rate must be consumed exactly once."
                ),
                mutation_guidance,
                (
                    "Return the smallest diagnostic-directed structural change; "
                    "family is only a label and never selects the executor."
                ),
            ],
            max_proposals=1,
            created_at=spec.created_at,
        )
        response = self.transport.propose(request)
        evidence = generation_call_evidence_v42(
            request, response, self.transport
        )
        if response.action == "stop":
            return []
        proposals: list[OpenEvolutionProposalV42] = []
        for index, draft in enumerate(response.proposals):
            operator = HybridEvolutionOperatorV42.seal(
                operator_id=(
                    f"generated.recursive.{draft.family}.g{next_generation}."
                    f"{candidate.candidate_hash[:8]}.{index}"
                ),
                channel="generated",
                kind=draft.kind,
                proposed_by="model",
                source_candidate_hash=candidate.candidate_hash,
                source_evaluation_hash=evaluation.evaluation_hash,
                failure_signature_hash=failure.failure_hash,
                target_family=draft.family,
                transformation_summary=draft.transformation_summary,
                rationale=draft.rationale,
            )
            proposals.append(
                OpenEvolutionProposalV42(
                    operator=operator,
                    candidate=_candidate_from_draft_v43(
                        draft,
                        source="generated",
                        parent=candidate,
                        operator=operator,
                        next_generation=next_generation,
                    ),
                    generation_evidence=evidence,
                    priority=draft.priority,
                )
            )
        return proposals


def recursive_event_process_campaign_spec_v43(
    development: USGSCatalogSnapshotV40,
    grammar: OpenModelGrammarV42,
    registry: TopologyCompilerRegistryV43,
    *,
    campaign_id: str,
    policy: EventProcessEvolutionPolicyV41 | None = None,
    max_generations: int = 3,
    max_candidates: int = 4,
) -> OpenEvolutionCampaignSpecV42:
    development.assert_sealed()
    grammar.assert_sealed()
    registry.assert_sealed()
    frozen_policy = policy or default_event_process_evolution_policy_v41()
    frozen_policy.assert_sealed()
    return OpenEvolutionCampaignSpecV42.seal(
        campaign_id=campaign_id,
        evaluator_epoch="event_process_recursive_development_v43",
        objective=(
            "Recursively evolve executable continuous-time event-process "
            "expressions through a declarative topology registry using only "
            "chronological development slices."
        ),
        development_data_hash=development.snapshot_hash,
        grammar_hash=grammar.grammar_hash,
        required_development_gates=EVENT_PROCESS_OPEN_DEVELOPMENT_GATES_V42,
        evaluation_policy={
            **frozen_policy.model_dump(
                mode="json", exclude={"schema_version", "policy_hash"}
            ),
            "policy_hash": frozen_policy.policy_hash,
            "topology_registry_hash": registry.registry_hash,
            "minimum_component_branching_ratio": 0.02,
            "minimum_adjacent_decay_ratio": 2.0,
        },
        max_generations=max_generations,
        max_candidates=max_candidates,
        prescribed_quota_per_failure=1,
        generated_quota_per_failure=1,
        created_at=development.retrieved_at,
    )


def run_recursive_event_process_campaign_v43(
    output_root,
    development: USGSCatalogSnapshotV40,
    transport: OpenEvolutionGenerationTransportV42,
    *,
    campaign_id: str,
    policy: EventProcessEvolutionPolicyV41 | None = None,
) -> OpenEvolutionOutcomeV42:
    grammar = event_process_recursive_grammar_v43()
    registry = event_process_topology_registry_v43()
    frozen_policy = policy or default_event_process_evolution_policy_v41()
    spec = recursive_event_process_campaign_spec_v43(
        development,
        grammar,
        registry,
        campaign_id=campaign_id,
        policy=frozen_policy,
    )
    adapter = RecursiveEventProcessEvolutionAdapterV43(
        development, transport, frozen_policy, registry
    )
    return run_open_evolution_campaign_v42(
        output_root, spec, grammar, adapter
    )
