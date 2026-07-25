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


EventProcessStructureV42 = Literal[
    "homogeneous_poisson",
    "weibull_renewal",
    "single_exponential_hawkes",
    "two_timescale_hawkes",
]


EVENT_PROCESS_OPEN_DEVELOPMENT_GATES_V42 = sorted(
    [
        "compensator_count_calibration",
        "component_identifiability",
        "development_slice_contract",
        "hawkes_stationarity",
        "optimizer_converged",
        "parameter_interior",
        "time_rescaling_calibration",
        "training_bic_not_worse",
        "validation_log_score_lift",
    ]
)


EVENT_PROCESS_OPEN_ADAPTER_ID_V42 = "event_process_open_v42"
EVENT_PROCESS_OPEN_COMPILER_HASH_V42 = sha256_value(
    {
        "compiler": "event_process_expression_topology_v42",
        "recognized_structures": [
            "homogeneous_poisson",
            "single_exponential_hawkes",
            "two_timescale_hawkes",
            "weibull_renewal",
        ],
        "terminal_output": "intensity",
        "all_applications_must_reach_terminal": True,
    }
)
EVENT_PROCESS_OPEN_ADAPTER_HASH_V42 = sha256_value(
    {
        "adapter": EVENT_PROCESS_OPEN_ADAPTER_ID_V42,
        "compiler_hash": EVENT_PROCESS_OPEN_COMPILER_HASH_V42,
        "fit": "maximum_likelihood_multistart_v42",
        "two_timescale_parameterization": (
            "background,total_branching,fast_fraction,fast_decay,slow_decay"
        ),
        "evaluation": "development_70_30_nine_frozen_gates_v42",
        "private_data_access": False,
    }
)


def _hash_without(model: StrictModel, field: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude={field}))


class CompiledEventProcessV42(StrictModel):
    schema_version: Literal["4.2"] = "4.2"
    candidate_hash: Sha256
    compiler_hash: Sha256
    structure: EventProcessStructureV42
    terminal_symbol: Literal["intensity"] = "intensity"
    application_ids: Annotated[list[Identifier], Field(min_length=1, max_length=32)]
    compiled_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_compiled(self) -> "CompiledEventProcessV42":
        if self.application_ids != sorted(set(self.application_ids)):
            raise ValueError("compiled application_ids must be sorted and unique")
        if self.compiled_hash and self.compiled_hash != self.content_hash():
            raise ValueError("compiled_hash does not match compiled event process")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "compiled_hash")

    def assert_sealed(self) -> None:
        if not self.compiled_hash or self.compiled_hash != self.content_hash():
            raise ValueError("compiled event process is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "CompiledEventProcessV42":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"compiled_hash"}),
            compiled_hash=draft.content_hash(),
        )


class FittedOpenEventProcessV42(StrictModel):
    schema_version: Literal["4.2"] = "4.2"
    candidate_hash: Sha256
    compiled_hash: Sha256
    development_snapshot_hash: Sha256
    structure: EventProcessStructureV42
    parameters: dict[
        Identifier, Annotated[float, Field(allow_inf_nan=False)]
    ]
    optimizer_converged: bool
    optimizer_message: Annotated[str, Field(min_length=1, max_length=2000)]
    development_log_likelihood: Annotated[
        float, Field(allow_inf_nan=False)
    ]
    development_bic: Annotated[float, Field(allow_inf_nan=False)]
    parameter_count: Annotated[int, Field(ge=1, le=32)]
    fit_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_fit(self) -> "FittedOpenEventProcessV42":
        if len(self.parameters) != self.parameter_count:
            raise ValueError("parameter_count differs from fitted parameters")
        if self.fit_hash and self.fit_hash != self.content_hash():
            raise ValueError("fit_hash does not match open event-process fit")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "fit_hash")

    def assert_sealed(self) -> None:
        if not self.fit_hash or self.fit_hash != self.content_hash():
            raise ValueError("open event-process fit is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "FittedOpenEventProcessV42":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"fit_hash"}),
            fit_hash=draft.content_hash(),
        )


def event_process_open_grammar_v42() -> OpenModelGrammarV42:
    return OpenModelGrammarV42.seal(
        grammar_id="event_process_expression_grammar_v42",
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
                max_uses_per_candidate=2,
            ),
            PrimitiveRuleV42(
                primitive_id="renewal_hazard",
                role="mechanism",
                input_units=["day", "unitless", "day"],
                output_unit="rate_per_day",
            ),
        ],
        executable_adapter_ids=[EVENT_PROCESS_OPEN_ADAPTER_ID_V42],
        executable_adapter_hashes={
            EVENT_PROCESS_OPEN_ADAPTER_ID_V42: (
                EVENT_PROCESS_OPEN_ADAPTER_HASH_V42
            )
        },
        forbidden_tokens=["private_confirmation", "qualification_granted"],
        max_symbols=20,
        max_applications=8,
    )


def _symbol_map(candidate: OpenModelCandidateV42) -> dict[str, ModelSymbolV42]:
    return {item.symbol_id: item for item in candidate.symbols}


def compile_event_process_candidate_v42(
    candidate: OpenModelCandidateV42,
) -> CompiledEventProcessV42:
    """Compile by expression topology, never by the model-supplied family label."""

    candidate.assert_sealed()
    symbols = _symbol_map(candidate)
    intensity = symbols.get("intensity")
    if intensity is None or intensity.role != "output" or intensity.unit != "rate_per_day":
        raise ValueError("event-process expression needs output intensity: rate_per_day")
    producers: dict[str, PrimitiveApplicationV42] = {}
    for application in candidate.applications:
        if application.output in producers:
            raise ValueError("event-process expression has duplicate producers")
        producers[application.output] = application
    if "intensity" not in producers:
        raise ValueError("event-process expression does not produce intensity")

    reachable_applications: set[str] = set()
    reachable_symbols: set[str] = set()

    def visit(symbol_id: str) -> None:
        if symbol_id in reachable_symbols:
            return
        reachable_symbols.add(symbol_id)
        producer = producers.get(symbol_id)
        if producer is None:
            return
        reachable_applications.add(producer.application_id)
        for input_symbol in producer.inputs:
            visit(input_symbol)

    visit("intensity")
    all_application_ids = {item.application_id for item in candidate.applications}
    if reachable_applications != all_application_ids:
        raise ValueError("event-process expression contains unused applications")
    counts: dict[str, int] = {}
    for application in candidate.applications:
        counts[application.primitive_id] = (
            counts.get(application.primitive_id, 0) + 1
        )
    if counts == {"constant_background": 1}:
        structure: EventProcessStructureV42 = "homogeneous_poisson"
    elif counts == {"renewal_hazard": 1}:
        structure = "weibull_renewal"
    elif counts == {
        "add_rate": 1,
        "constant_background": 1,
        "exponential_memory": 1,
    }:
        structure = "single_exponential_hawkes"
    elif counts == {
        "add_rate": 2,
        "constant_background": 1,
        "exponential_memory": 2,
    }:
        structure = "two_timescale_hawkes"
    else:
        raise ValueError("event-process expression topology is not executable")
    return CompiledEventProcessV42.seal(
        candidate_hash=candidate.candidate_hash,
        compiler_hash=EVENT_PROCESS_OPEN_COMPILER_HASH_V42,
        structure=structure,
        application_ids=sorted(all_application_ids),
    )


def _single_hawkes_candidate_v42() -> OpenModelCandidateV42:
    return OpenModelCandidateV42.seal(
        candidate_id="event.open.single_hawkes.g0",
        generation=0,
        family="single_exponential_hawkes",
        source="seed",
        proposed_by="model",
        executable_adapter_id=EVENT_PROCESS_OPEN_ADAPTER_ID_V42,
        symbols=[
            ModelSymbolV42(
                symbol_id="background",
                role="parameter",
                unit="rate_per_day",
            ),
            ModelSymbolV42(
                symbol_id="background_rate",
                role="derived",
                unit="rate_per_day",
            ),
            ModelSymbolV42(
                symbol_id="branching",
                role="parameter",
                unit="unitless",
            ),
            ModelSymbolV42(
                symbol_id="decay",
                role="parameter",
                unit="rate_per_day",
            ),
            ModelSymbolV42(
                symbol_id="event_history",
                role="observed",
                unit="event_history",
            ),
            ModelSymbolV42(
                symbol_id="history_rate",
                role="derived",
                unit="rate_per_day",
            ),
            ModelSymbolV42(
                symbol_id="intensity",
                role="output",
                unit="rate_per_day",
            ),
        ],
        applications=[
            PrimitiveApplicationV42(
                application_id="background_component",
                primitive_id="constant_background",
                inputs=["background"],
                output="background_rate",
            ),
            PrimitiveApplicationV42(
                application_id="history_component",
                primitive_id="exponential_memory",
                inputs=["event_history", "branching", "decay"],
                output="history_rate",
            ),
            PrimitiveApplicationV42(
                application_id="intensity_sum",
                primitive_id="add_rate",
                inputs=["background_rate", "history_rate"],
                output="intensity",
            ),
        ],
        assumptions=[
            "Stationary self-excitation has one exponential memory timescale.",
            "All development events share one background intensity.",
        ],
        rationale=(
            "Begin from a compact self-exciting point process, then let frozen "
            "development diagnostics justify any structural expansion."
        ),
        expected_failure_modes=[
            "A single decay component may collapse to a boundary.",
            "One timescale may miss both rapid and persistent triggering.",
        ],
    )


def _renewal_draft_v42() -> GeneratedModelDraftV42:
    return GeneratedModelDraftV42(
        family="weibull_renewal",
        kind="replace_skeleton",
        symbols=[
            ModelSymbolV42(
                symbol_id="elapsed_time",
                role="state",
                unit="day",
            ),
            ModelSymbolV42(
                symbol_id="intensity",
                role="output",
                unit="rate_per_day",
            ),
            ModelSymbolV42(
                symbol_id="scale",
                role="parameter",
                unit="day",
            ),
            ModelSymbolV42(
                symbol_id="shape",
                role="parameter",
                unit="unitless",
            ),
        ],
        applications=[
            PrimitiveApplicationV42(
                application_id="renewal_intensity",
                primitive_id="renewal_hazard",
                inputs=["elapsed_time", "shape", "scale"],
                output="intensity",
            )
        ],
        assumptions=[
            "Waiting-time dependence resets after each observed event.",
            "Interarrival times follow one Weibull renewal law.",
        ],
        transformation_summary=(
            "Replace branching memory with a renewal hazard driven by elapsed time."
        ),
        rationale=(
            "A renewal skeleton is a prescribed low-complexity alternative when "
            "the self-exciting fit fails frozen development gates."
        ),
        expected_failure_modes=[
            "Renewal memory may miss branching clusters.",
            "One interarrival law may miss temporal rate drift.",
        ],
        priority=0.5,
    )


def _candidate_from_draft(
    draft: GeneratedModelDraftV42,
    *,
    source: Literal["prescribed", "generated"],
    parent: OpenModelCandidateV42,
    operator: HybridEvolutionOperatorV42,
    next_generation: int,
) -> OpenModelCandidateV42:
    return OpenModelCandidateV42.seal(
        candidate_id=(
            f"event.open.{draft.family}.g{next_generation}."
            f"{parent.candidate_hash[:8]}"
        ),
        generation=next_generation,
        family=draft.family,
        source=source,
        proposed_by="harness" if source == "prescribed" else "model",
        executable_adapter_id=EVENT_PROCESS_OPEN_ADAPTER_ID_V42,
        symbols=draft.symbols,
        applications=draft.applications,
        assumptions=draft.assumptions,
        parent_candidate_hashes=[parent.candidate_hash],
        operator_hashes=[operator.operator_hash],
        rationale=draft.rationale,
        expected_failure_modes=draft.expected_failure_modes,
    )


def _two_timescale_components(
    events: np.ndarray,
    *,
    start: float,
    end: float,
    history: np.ndarray,
    background: float,
    total_branching: float,
    fast_fraction: float,
    fast_decay: float,
    slow_decay: float,
) -> tuple[float, list[float], float]:
    if (
        background <= 0
        or not 0 <= total_branching < 1
        or not 0 <= fast_fraction <= 1
        or fast_decay <= slow_decay
        or slow_decay <= 0
    ):
        return -math.inf, [], math.inf
    fast_branching = total_branching * fast_fraction
    slow_branching = total_branching * (1.0 - fast_fraction)
    prior = [float(value) for value in history if value < start]
    log_terms: list[float] = []
    rescaled: list[float] = []
    left = start
    total_compensator = 0.0

    def kernel_intensity(event_time: float) -> float:
        return sum(
            fast_branching
            * fast_decay
            * math.exp(-fast_decay * (event_time - past))
            + slow_branching
            * slow_decay
            * math.exp(-slow_decay * (event_time - past))
            for past in prior
            if past < event_time
        )

    def kernel_integral(interval_start: float, interval_end: float) -> float:
        return sum(
            fast_branching
            * (
                math.exp(-fast_decay * max(interval_start - past, 0.0))
                - math.exp(-fast_decay * (interval_end - past))
            )
            + slow_branching
            * (
                math.exp(-slow_decay * max(interval_start - past, 0.0))
                - math.exp(-slow_decay * (interval_end - past))
            )
            for past in prior
            if past <= interval_start
        )

    for event in events:
        event_time = float(event)
        intensity = background + kernel_intensity(event_time)
        if not math.isfinite(intensity) or intensity <= 0:
            return -math.inf, [], math.inf
        interval_integral = (
            background * (event_time - left)
            + kernel_integral(left, event_time)
        )
        if not math.isfinite(interval_integral) or interval_integral < 0:
            return -math.inf, [], math.inf
        log_terms.append(math.log(intensity))
        rescaled.append(interval_integral)
        total_compensator += interval_integral
        prior.append(event_time)
        left = event_time
    tail = background * (end - left) + kernel_integral(left, end)
    total_compensator += tail
    return sum(log_terms) - total_compensator, rescaled, total_compensator


def fit_open_event_process_v42(
    candidate: OpenModelCandidateV42,
    development: USGSCatalogSnapshotV40,
) -> tuple[CompiledEventProcessV42, FittedOpenEventProcessV42]:
    candidate.assert_sealed()
    development.assert_sealed()
    compiled = compile_event_process_candidate_v42(candidate)
    structure = compiled.structure
    if structure != "two_timescale_hawkes":
        family = {
            "homogeneous_poisson": "homogeneous_poisson",
            "weibull_renewal": "weibull_renewal",
            "single_exponential_hawkes": "exponential_hawkes",
        }[structure]
        domain_candidate = EventProcessCandidateV40.seal(
            family=family,
            hawkes_branching_initial=0.45,
            hawkes_decay_days_initial=2.0,
            rationale=(
                "Compile the admitted V4.2 expression into the existing "
                "maximum-likelihood event-process implementation."
            ),
            expected_failure_modes=[
                "The compiled structure may fail frozen development gates."
            ],
        )
        domain_fit = fit_event_process_v40(domain_candidate, development)
        fit = FittedOpenEventProcessV42.seal(
            candidate_hash=candidate.candidate_hash,
            compiled_hash=compiled.compiled_hash,
            development_snapshot_hash=development.snapshot_hash,
            structure=structure,
            parameters=domain_fit.parameters,
            optimizer_converged=domain_fit.optimizer_converged,
            optimizer_message=domain_fit.optimizer_message,
            development_log_likelihood=domain_fit.development_log_likelihood,
            development_bic=domain_fit.development_bic,
            parameter_count=domain_fit.parameter_count,
        )
        return compiled, fit

    events = _relative_times(development)
    duration = _duration_days(development.query)
    count = len(events)
    rate = count / duration
    upper_background = max(10.0, 10.0 * rate)
    starts = [
        [max(rate * 0.55, 1e-4), 0.45, 0.60, 1.0, 1.0 / 14.0],
        [max(rate * 0.75, 1e-4), 0.25, 0.50, 2.0, 1.0 / 30.0],
        [max(rate * 0.35, 1e-4), 0.65, 0.75, 0.5, 1.0 / 7.0],
    ]

    def objective(values: np.ndarray) -> float:
        likelihood, _, _ = _two_timescale_components(
            events,
            start=0.0,
            end=duration,
            history=np.asarray([], dtype=float),
            background=float(values[0]),
            total_branching=float(values[1]),
            fast_fraction=float(values[2]),
            fast_decay=float(values[3]),
            slow_decay=float(values[4]),
        )
        return -likelihood if math.isfinite(likelihood) else 1e100

    results = [
        minimize(
            objective,
            np.asarray(start, dtype=float),
            method="L-BFGS-B",
            bounds=[
                (1e-6, upper_background),
                (1e-6, 0.949999),
                (0.02, 0.98),
                (0.2001, 10.0),
                (1.0 / 60.0, 0.1999),
            ],
        )
        for start in starts
    ]
    result = min(results, key=lambda item: float(item.fun))
    log_likelihood = -float(result.fun)
    total_branching = float(result.x[1])
    fast_fraction = float(result.x[2])
    parameters = {
        "background_rate_per_day": float(result.x[0]),
        "total_branching_ratio": total_branching,
        "fast_fraction": fast_fraction,
        "fast_decay_rate_per_day": float(result.x[3]),
        "slow_decay_rate_per_day": float(result.x[4]),
    }
    parameter_count = 5
    bic = parameter_count * math.log(count) - 2.0 * log_likelihood
    fit = FittedOpenEventProcessV42.seal(
        candidate_hash=candidate.candidate_hash,
        compiled_hash=compiled.compiled_hash,
        development_snapshot_hash=development.snapshot_hash,
        structure=structure,
        parameters=parameters,
        optimizer_converged=bool(
            result.success and math.isfinite(log_likelihood)
        ),
        optimizer_message=str(result.message),
        development_log_likelihood=log_likelihood,
        development_bic=bic,
        parameter_count=len(parameters),
    )
    return compiled, fit


def _validation_components_v42(
    fit: FittedOpenEventProcessV42,
    training: USGSCatalogSnapshotV40,
    validation: USGSCatalogSnapshotV40,
) -> tuple[float, list[float], float]:
    if fit.structure != "two_timescale_hawkes":
        family = {
            "homogeneous_poisson": "homogeneous_poisson",
            "weibull_renewal": "weibull_renewal",
            "single_exponential_hawkes": "exponential_hawkes",
        }[fit.structure]
        domain_fit = FittedEventProcessV40.seal(
            candidate_hash=fit.candidate_hash,
            development_snapshot_hash=training.snapshot_hash,
            family=family,
            parameters=fit.parameters,
            optimizer_converged=fit.optimizer_converged,
            optimizer_message=fit.optimizer_message,
            development_log_likelihood=fit.development_log_likelihood,
            development_bic=fit.development_bic,
            parameter_count=fit.parameter_count,
        )
        return _confirmation_components(domain_fit, training, validation)
    events = _relative_times(validation)
    history = np.asarray(
        [
            (event.origin_time - validation.query.start).total_seconds()
            / 86400.0
            for event in training.events
        ],
        dtype=float,
    )
    return _two_timescale_components(
        events,
        start=0.0,
        end=_duration_days(validation.query),
        history=history,
        background=fit.parameters["background_rate_per_day"],
        total_branching=fit.parameters["total_branching_ratio"],
        fast_fraction=fit.parameters["fast_fraction"],
        fast_decay=fit.parameters["fast_decay_rate_per_day"],
        slow_decay=fit.parameters["slow_decay_rate_per_day"],
    )


class EventProcessOpenEvolutionAdapterV42:
    """Real development-only adapter with typed generated structure transport."""

    adapter_id = EVENT_PROCESS_OPEN_ADAPTER_ID_V42
    adapter_contract_hash = EVENT_PROCESS_OPEN_ADAPTER_HASH_V42

    def __init__(
        self,
        development: USGSCatalogSnapshotV40,
        transport: OpenEvolutionGenerationTransportV42,
        policy: EventProcessEvolutionPolicyV41 | None = None,
    ) -> None:
        development.assert_sealed()
        self.development = development
        self.transport = transport
        self.policy = policy or default_event_process_evolution_policy_v41()
        self.policy.assert_sealed()
        self.training, self.validation = split_event_process_development_v41(
            development, self.policy.split_fraction
        )

    def _assert_spec(
        self,
        spec: OpenEvolutionCampaignSpecV42,
        grammar: OpenModelGrammarV42 | None = None,
    ) -> None:
        if spec.development_data_hash != self.development.snapshot_hash:
            raise ValueError("campaign refers to another development snapshot")
        if spec.required_development_gates != EVENT_PROCESS_OPEN_DEVELOPMENT_GATES_V42:
            raise ValueError("campaign gate set differs from V4.2 event policy")
        if spec.evaluation_policy.get("policy_hash") != self.policy.policy_hash:
            raise ValueError("campaign policy hash differs from adapter policy")
        if grammar is not None and grammar.grammar_hash != spec.grammar_hash:
            raise ValueError("campaign refers to another model grammar")

    def initial_candidates(
        self,
        spec: OpenEvolutionCampaignSpecV42,
        grammar: OpenModelGrammarV42,
    ) -> list[OpenModelCandidateV42]:
        self._assert_spec(spec, grammar)
        return [_single_hawkes_candidate_v42()]

    def supports_candidate(self, candidate: OpenModelCandidateV42) -> bool:
        try:
            compiled = compile_event_process_candidate_v42(candidate)
            return (
                compiled.compiler_hash == EVENT_PROCESS_OPEN_COMPILER_HASH_V42
            )
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
            raise ValueError("execution attempt is bound to another candidate")
        compiled, fit = fit_open_event_process_v42(candidate, self.training)
        return DevelopmentExecutionV42.seal(
            candidate_hash=candidate.candidate_hash,
            attempt_hash=attempt.attempt_hash,
            idempotency_key=attempt.idempotency_key,
            development_data_hash=self.development.snapshot_hash,
            converged=fit.optimizer_converged,
            metrics={
                "training_bic": fit.development_bic,
                "training_log_likelihood": fit.development_log_likelihood,
                "parameter_count": float(fit.parameter_count),
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
            raise ValueError("development execution lacks an event-process fit")
        fit = FittedOpenEventProcessV42.model_validate(fit_payload)
        fit.assert_sealed()
        if fit.candidate_hash != candidate.candidate_hash:
            raise ValueError("fit belongs to another open model candidate")

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
            _validation_components_v42(fit, self.training, self.validation)
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

        if fit.structure == "two_timescale_hawkes":
            branching = fit.parameters["total_branching_ratio"]
            decay_ratio = (
                fit.parameters["fast_decay_rate_per_day"]
                / fit.parameters["slow_decay_rate_per_day"]
            )
            component_identifiability = (
                branching * fit.parameters["fast_fraction"] >= 0.02
                and branching * (1.0 - fit.parameters["fast_fraction"]) >= 0.02
                and decay_ratio >= 2.0
            )
            parameter_interior = (
                fit.parameters["fast_decay_rate_per_day"]
                < self.policy.maximum_interior_hawkes_decay_rate_per_day
                and fit.parameters["slow_decay_rate_per_day"] > 1.0 / 59.99
                and 0.021 < fit.parameters["fast_fraction"] < 0.979
            )
        elif fit.structure == "single_exponential_hawkes":
            branching = fit.parameters["branching_ratio"]
            decay_ratio = 1.0
            component_identifiability = True
            parameter_interior = (
                fit.parameters["decay_rate_per_day"]
                < self.policy.maximum_interior_hawkes_decay_rate_per_day
            )
        elif fit.structure == "weibull_renewal":
            branching = 0.0
            decay_ratio = 0.0
            component_identifiability = True
            parameter_interior = (
                0.1001 < fit.parameters["shape"] < 9.999
                and 0.00011 < fit.parameters["scale_days"] < 99.99
            )
        else:
            branching = 0.0
            decay_ratio = 0.0
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
                fit.structure
                not in {"single_exponential_hawkes", "two_timescale_hawkes"}
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
        diagnostics: list[str] = []
        diagnostic_by_gate = {
            "optimizer_converged": "optimizer_failure",
            "training_bic_not_worse": "unsupported_complexity",
            "validation_log_score_lift": "no_predictive_lift",
            "time_rescaling_calibration": "time_rescaling_miscalibration",
            "parameter_interior": "parameter_boundary",
            "hawkes_stationarity": "hawkes_nonstationarity",
            "development_slice_contract": "insufficient_development_slice",
            "component_identifiability": "components_not_identifiable",
        }
        diagnostics.extend(
            code for gate, code in diagnostic_by_gate.items() if not gates[gate]
        )
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
                "decay_ratio": decay_ratio,
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
        if (
            evaluation.disposition == "discard"
            or spec.generated_quota_per_failure == 0
        ):
            return []
        draft = _renewal_draft_v42()
        operator = HybridEvolutionOperatorV42.seal(
            operator_id=(
                f"prescribed.renewal.g{next_generation}."
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
                candidate=_candidate_from_draft(
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
        if evaluation.disposition == "discard":
            return []
        request = OpenEvolutionGenerationRequestV42.seal(
            request_id=(
                f"generate.{candidate.candidate_hash[:12]}."
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
                "Produce an acyclic expression whose final output symbol is intensity.",
                "Every application must contribute to intensity; unused branches fail compilation.",
                "Distinct exponential-memory components may use distinct parameters and be summed.",
                "Prefer the smallest structural change that addresses the supplied diagnostics.",
            ],
            max_proposals=spec.generated_quota_per_failure,
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
                    f"generated.{draft.family}.g{next_generation}."
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
                    candidate=_candidate_from_draft(
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


def event_process_open_campaign_spec_v42(
    development: USGSCatalogSnapshotV40,
    grammar: OpenModelGrammarV42,
    *,
    campaign_id: str,
    policy: EventProcessEvolutionPolicyV41 | None = None,
    max_generations: int = 2,
    max_candidates: int = 4,
    prescribed_quota_per_failure: int = 1,
    generated_quota_per_failure: int = 2,
) -> OpenEvolutionCampaignSpecV42:
    development.assert_sealed()
    grammar.assert_sealed()
    frozen_policy = policy or default_event_process_evolution_policy_v41()
    frozen_policy.assert_sealed()
    return OpenEvolutionCampaignSpecV42.seal(
        campaign_id=campaign_id,
        evaluator_epoch="event_process_open_development_v42",
        objective=(
            "Discover executable continuous-time event-process structures using "
            "only chronological development slices, then freeze an unqualified "
            "development champion."
        ),
        development_data_hash=development.snapshot_hash,
        grammar_hash=grammar.grammar_hash,
        required_development_gates=EVENT_PROCESS_OPEN_DEVELOPMENT_GATES_V42,
        evaluation_policy={
            **frozen_policy.model_dump(
                mode="json", exclude={"schema_version", "policy_hash"}
            ),
            "policy_hash": frozen_policy.policy_hash,
            "minimum_component_branching_ratio": 0.02,
            "minimum_component_decay_ratio": 2.0,
        },
        max_generations=max_generations,
        max_candidates=max_candidates,
        prescribed_quota_per_failure=prescribed_quota_per_failure,
        generated_quota_per_failure=generated_quota_per_failure,
        created_at=development.retrieved_at,
    )


def run_event_process_open_campaign_v42(
    output_root,
    development: USGSCatalogSnapshotV40,
    transport: OpenEvolutionGenerationTransportV42,
    *,
    campaign_id: str,
    policy: EventProcessEvolutionPolicyV41 | None = None,
) -> OpenEvolutionOutcomeV42:
    grammar = event_process_open_grammar_v42()
    frozen_policy = policy or default_event_process_evolution_policy_v41()
    spec = event_process_open_campaign_spec_v42(
        development,
        grammar,
        campaign_id=campaign_id,
        policy=frozen_policy,
    )
    adapter = EventProcessOpenEvolutionAdapterV42(
        development, transport, frozen_policy
    )
    return run_open_evolution_campaign_v42(
        output_root, spec, grammar, adapter
    )
