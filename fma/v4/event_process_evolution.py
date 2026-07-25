from __future__ import annotations

import math
from datetime import timedelta
from typing import Annotated, Literal

import numpy as np
from pydantic import Field, model_validator
from scipy.stats import kstest

from fma.hashing import sha256_value
from fma.schemas import StrictModel
from fma.v2.schemas import Sha256

from .model_evolution_graph import (
    DevelopmentEvaluationV41,
    DevelopmentExecutionV41,
    EvolutionOperatorV41,
    EvolutionProposalV41,
    FailureSignatureV41,
    ModelCandidateV41,
    ModelEvolutionAdapterV41,
    ModelEvolutionCampaignSpecV41,
    ModelEvolutionOutcomeV41,
    run_model_evolution_campaign_v41,
)
from .unseen_event_process import (
    EventProcessCandidateV40,
    FittedEventProcessV40,
    USGSCatalogQueryV40,
    USGSCatalogSnapshotV40,
    _confirmation_components,
    _duration_days,
    _poisson_log_likelihood,
    fit_event_process_v40,
)


EVENT_PROCESS_DEVELOPMENT_GATES_V41 = sorted(
    [
        "compensator_count_calibration",
        "development_slice_contract",
        "hawkes_stationarity",
        "optimizer_converged",
        "parameter_interior",
        "time_rescaling_calibration",
        "training_bic_not_worse",
        "validation_log_score_lift",
    ]
)


def _hash_without(model: StrictModel, field: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude={field}))


class EventProcessEvolutionPolicyV41(StrictModel):
    schema_version: Literal["4.1"] = "4.1"
    split_fraction: Annotated[float, Field(gt=0.5, lt=0.9, allow_inf_nan=False)] = 0.7
    minimum_events_per_slice: Annotated[int, Field(ge=5)] = 20
    minimum_validation_log_score_lift_nat_per_event: Annotated[
        float, Field(allow_inf_nan=False)
    ] = 0.0
    minimum_time_rescaling_ks_pvalue: Annotated[
        float, Field(gt=0, lt=1, allow_inf_nan=False)
    ] = 0.01
    maximum_compensator_count_relative_error: Annotated[
        float, Field(gt=0, lt=1, allow_inf_nan=False)
    ] = 0.25
    maximum_hawkes_branching_ratio: Annotated[
        float, Field(gt=0, lt=1, allow_inf_nan=False)
    ] = 0.95
    maximum_interior_hawkes_decay_rate_per_day: Annotated[
        float, Field(gt=0, lt=10, allow_inf_nan=False)
    ] = 9.95
    policy_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_policy(self) -> "EventProcessEvolutionPolicyV41":
        if self.policy_hash and self.policy_hash != self.content_hash():
            raise ValueError("policy_hash does not match event-process evolution policy")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "policy_hash")

    def assert_sealed(self) -> None:
        if not self.policy_hash or self.policy_hash != self.content_hash():
            raise ValueError("event-process evolution policy is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "EventProcessEvolutionPolicyV41":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"policy_hash"}),
            policy_hash=draft.content_hash(),
        )


def default_event_process_evolution_policy_v41() -> EventProcessEvolutionPolicyV41:
    return EventProcessEvolutionPolicyV41.seal()


def event_process_evolution_campaign_spec_v41(
    development: USGSCatalogSnapshotV40,
    *,
    campaign_id: str,
    policy: EventProcessEvolutionPolicyV41 | None = None,
    max_generations: int = 2,
    beam_width: int = 2,
    max_candidates: int = 3,
) -> ModelEvolutionCampaignSpecV41:
    development.assert_sealed()
    frozen_policy = policy or default_event_process_evolution_policy_v41()
    frozen_policy.assert_sealed()
    return ModelEvolutionCampaignSpecV41.seal(
        campaign_id=campaign_id,
        evaluator_epoch="event_process_development_v41",
        objective=(
            "Evolve continuous event-time model families on development-only "
            "chronological slices and freeze an unqualified champion."
        ),
        development_data_hash=development.snapshot_hash,
        required_gates=EVENT_PROCESS_DEVELOPMENT_GATES_V41,
        evaluation_policy={
            **frozen_policy.model_dump(
                mode="json", exclude={"schema_version", "policy_hash"}
            ),
            "policy_hash": frozen_policy.policy_hash,
        },
        max_generations=max_generations,
        beam_width=beam_width,
        max_candidates=max_candidates,
        created_at=development.retrieved_at,
    )


def _query_slice(
    source: USGSCatalogQueryV40,
    *,
    start,
    end_exclusive,
) -> USGSCatalogQueryV40:
    return USGSCatalogQueryV40.seal(
        phase="development",
        start=start,
        end_exclusive=end_exclusive,
        catalog=source.catalog,
        event_type=source.event_type,
        min_latitude=source.min_latitude,
        max_latitude=source.max_latitude,
        min_longitude=source.min_longitude,
        max_longitude=source.max_longitude,
        min_magnitude=source.min_magnitude,
        max_depth_km=source.max_depth_km,
    )


def split_event_process_development_v41(
    development: USGSCatalogSnapshotV40,
    split_fraction: float,
) -> tuple[USGSCatalogSnapshotV40, USGSCatalogSnapshotV40]:
    development.assert_sealed()
    duration = development.query.end_exclusive - development.query.start
    split = development.query.start + timedelta(
        seconds=duration.total_seconds() * split_fraction
    )
    training_events = [
        event for event in development.events if event.origin_time < split
    ]
    validation_events = [
        event for event in development.events if event.origin_time >= split
    ]
    if not training_events or not validation_events:
        raise ValueError("development split needs events on both sides")
    training = USGSCatalogSnapshotV40.seal(
        query=_query_slice(
            development.query,
            start=development.query.start,
            end_exclusive=split,
        ),
        response_sha256=development.response_sha256,
        events=training_events,
        retrieved_at=development.retrieved_at,
    )
    validation = USGSCatalogSnapshotV40.seal(
        query=_query_slice(
            development.query,
            start=split,
            end_exclusive=development.query.end_exclusive,
        ),
        response_sha256=development.response_sha256,
        events=validation_events,
        retrieved_at=development.retrieved_at,
    )
    return training, validation


def _domain_candidate(candidate: ModelCandidateV41) -> EventProcessCandidateV40:
    if candidate.model_spec.get("kind") != "event_process_candidate_v40":
        raise ValueError("model-evolution candidate is not an event process")
    payload = candidate.model_spec.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("event-process candidate payload is invalid")
    domain_candidate = EventProcessCandidateV40.model_validate(payload)
    domain_candidate.assert_sealed()
    if domain_candidate.family != candidate.family:
        raise ValueError("event-process family differs from graph candidate")
    return domain_candidate


def _graph_candidate(
    domain_candidate: EventProcessCandidateV40,
    *,
    generation: int,
    rationale: str,
    parent_candidate_hash: str | None = None,
    operator_hash: str | None = None,
) -> ModelCandidateV41:
    domain_candidate.assert_sealed()
    return ModelCandidateV41.seal(
        candidate_id=f"event_process.{domain_candidate.family}.g{generation}",
        generation=generation,
        family=domain_candidate.family,
        model_spec={
            "kind": "event_process_candidate_v40",
            "payload": domain_candidate.model_dump(mode="json"),
        },
        parent_candidate_hashes=(
            [parent_candidate_hash] if parent_candidate_hash else []
        ),
        operator_hashes=[operator_hash] if operator_hash else [],
        rationale=rationale,
        expected_failure_modes={
            "homogeneous_poisson": [
                "constant intensity may miss clustering and rate drift"
            ],
            "weibull_renewal": [
                "renewal memory may miss branching aftershock structure"
            ],
            "exponential_hawkes": [
                "one exponential kernel may miss multiple triggering timescales"
            ],
        }[domain_candidate.family],
    )


def _family_candidate(family: str) -> EventProcessCandidateV40:
    return EventProcessCandidateV40.seal(
        family=family,
        hawkes_branching_initial=0.45,
        hawkes_decay_days_initial=2.0,
        rationale=(
            f"Use {family} as a typed development candidate selected from a "
            "failure-driven model-family graph."
        ),
        expected_failure_modes={
            "homogeneous_poisson": [
                "constant rate may fail under clustering or temporal drift"
            ],
            "weibull_renewal": [
                "renewal dependence may not represent self-exciting aftershocks"
            ],
            "exponential_hawkes": [
                "a single decay timescale may be structurally insufficient"
            ],
        }[family],
    )


class EventProcessEvolutionAdapterV41(ModelEvolutionAdapterV41):
    """Development-only adapter; it never receives a confirmation snapshot."""

    def __init__(
        self,
        development: USGSCatalogSnapshotV40,
        policy: EventProcessEvolutionPolicyV41 | None = None,
        *,
        initial_family: Literal[
            "homogeneous_poisson", "weibull_renewal", "exponential_hawkes"
        ] = "exponential_hawkes",
    ) -> None:
        development.assert_sealed()
        self.development = development
        self.policy = policy or default_event_process_evolution_policy_v41()
        self.policy.assert_sealed()
        self.initial_family = initial_family
        self.training, self.validation = split_event_process_development_v41(
            development, self.policy.split_fraction
        )

    def _assert_spec(self, spec: ModelEvolutionCampaignSpecV41) -> None:
        if spec.development_data_hash != self.development.snapshot_hash:
            raise ValueError("campaign refers to another development snapshot")
        if spec.required_gates != EVENT_PROCESS_DEVELOPMENT_GATES_V41:
            raise ValueError("campaign gate set differs from event-process policy")
        if spec.evaluation_policy.get("policy_hash") != self.policy.policy_hash:
            raise ValueError("campaign policy hash differs from adapter policy")

    def initial_candidates(
        self, spec: ModelEvolutionCampaignSpecV41
    ) -> list[ModelCandidateV41]:
        self._assert_spec(spec)
        domain_candidate = _family_candidate(self.initial_family)
        return [
            _graph_candidate(
                domain_candidate,
                generation=0,
                rationale=(
                    "Start from the previous event-process direction, then let "
                    "development diagnostics determine whether to switch skeletons."
                ),
            )
        ]

    def execute(
        self,
        spec: ModelEvolutionCampaignSpecV41,
        candidate: ModelCandidateV41,
    ) -> DevelopmentExecutionV41:
        self._assert_spec(spec)
        domain_candidate = _domain_candidate(candidate)
        fit = fit_event_process_v40(domain_candidate, self.training)
        return DevelopmentExecutionV41.seal(
            candidate_hash=candidate.candidate_hash,
            development_data_hash=self.development.snapshot_hash,
            converged=fit.optimizer_converged,
            metrics={
                "training_bic": fit.development_bic,
                "training_log_likelihood": fit.development_log_likelihood,
                "parameter_count": float(fit.parameter_count),
            },
            domain_payload={
                "fit": fit.model_dump(mode="json"),
                "training_snapshot_hash": self.training.snapshot_hash,
                "validation_snapshot_hash": self.validation.snapshot_hash,
                "training_event_count": len(self.training.events),
                "validation_event_count": len(self.validation.events),
            },
        )

    def evaluate(
        self,
        spec: ModelEvolutionCampaignSpecV41,
        candidate: ModelCandidateV41,
        execution: DevelopmentExecutionV41,
    ) -> DevelopmentEvaluationV41:
        self._assert_spec(spec)
        domain_candidate = _domain_candidate(candidate)
        fit_payload = execution.domain_payload.get("fit")
        if not isinstance(fit_payload, dict):
            raise ValueError("development execution lacks a fitted event process")
        fit = FittedEventProcessV40.model_validate(fit_payload)
        fit.assert_sealed()
        if fit.candidate_hash != domain_candidate.candidate_hash:
            raise ValueError("fitted event process belongs to another domain candidate")

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
            baseline_rate,
            len(self.validation.events),
            validation_duration,
        )
        candidate_validation_ll, rescaled, compensator = _confirmation_components(
            fit, self.training, self.validation
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
        branching = fit.parameters.get("branching_ratio", 0.0)
        if fit.family == "exponential_hawkes":
            parameter_interior = (
                fit.parameters["decay_rate_per_day"]
                < self.policy.maximum_interior_hawkes_decay_rate_per_day
            )
        elif fit.family == "weibull_renewal":
            parameter_interior = (
                0.1001 < fit.parameters["shape"] < 9.999
                and 0.00011 < fit.parameters["scale_days"] < 99.99
            )
        else:
            parameter_interior = True
        gates = {
            "compensator_count_calibration": (
                count_error
                <= self.policy.maximum_compensator_count_relative_error
            ),
            "development_slice_contract": (
                len(self.training.events) >= self.policy.minimum_events_per_slice
                and len(self.validation.events)
                >= self.policy.minimum_events_per_slice
            ),
            "hawkes_stationarity": (
                fit.family != "exponential_hawkes"
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
        if not gates["optimizer_converged"]:
            diagnostics.append("optimizer_failure")
        if not gates["training_bic_not_worse"]:
            diagnostics.append("unsupported_complexity")
        if not gates["validation_log_score_lift"]:
            diagnostics.append("no_predictive_lift")
        if not gates["time_rescaling_calibration"]:
            diagnostics.append("time_rescaling_miscalibration")
        if not gates["compensator_count_calibration"]:
            diagnostics.append(
                "count_underprediction"
                if compensator < validation_count
                else "count_overprediction"
            )
        if not gates["parameter_interior"]:
            diagnostics.append("parameter_boundary")
        if not gates["hawkes_stationarity"]:
            diagnostics.append("hawkes_nonstationarity")
        if not gates["development_slice_contract"]:
            diagnostics.append("insufficient_development_slice")

        ks_term = 0.05 * math.log10(
            max(ks_pvalue, 1e-15)
            / self.policy.minimum_time_rescaling_ks_pvalue
        )
        failed_gate_penalty = 0.05 * sum(not value for value in gates.values())
        utility = log_score_lift - count_error + ks_term - failed_gate_penalty
        return DevelopmentEvaluationV41.seal(
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

    def evolve(
        self,
        spec: ModelEvolutionCampaignSpecV41,
        candidate: ModelCandidateV41,
        evaluation: DevelopmentEvaluationV41,
        failure: FailureSignatureV41,
        next_generation: int,
    ) -> list[EvolutionProposalV41]:
        self._assert_spec(spec)
        if evaluation.disposition == "discard":
            return []
        alternatives = {
            "exponential_hawkes": ["weibull_renewal", "homogeneous_poisson"],
            "weibull_renewal": ["exponential_hawkes", "homogeneous_poisson"],
            "homogeneous_poisson": ["exponential_hawkes", "weibull_renewal"],
        }[candidate.family]
        proposals: list[EvolutionProposalV41] = []
        for rank, family in enumerate(alternatives):
            operator = EvolutionOperatorV41.seal(
                operator_id=(
                    f"replace.{candidate.family}.{family}."
                    f"g{next_generation}.{candidate.candidate_hash[:8]}"
                ),
                kind="replace_skeleton",
                source_candidate_hash=candidate.candidate_hash,
                source_evaluation_hash=evaluation.evaluation_hash,
                failure_signature_hash=failure.failure_hash,
                diagnostic_codes=failure.diagnostic_codes,
                target_family=family,
                rationale=(
                    f"Switch from {candidate.family} to {family}; the new family "
                    "must independently pass every frozen development gate."
                ),
            )
            child = _graph_candidate(
                _family_candidate(family),
                generation=next_generation,
                parent_candidate_hash=candidate.candidate_hash,
                operator_hash=operator.operator_hash,
                rationale=(
                    f"Use verifier diagnostics from {candidate.family} to explore "
                    f"the structurally distinct {family} family."
                ),
            )
            proposals.append(
                EvolutionProposalV41(
                    operator=operator,
                    candidate=child,
                    priority=1.0 - rank * 0.1,
                )
            )
        return proposals


def run_event_process_evolution_campaign_v41(
    output_root,
    development: USGSCatalogSnapshotV40,
    *,
    campaign_id: str,
    policy: EventProcessEvolutionPolicyV41 | None = None,
) -> ModelEvolutionOutcomeV41:
    frozen_policy = policy or default_event_process_evolution_policy_v41()
    spec = event_process_evolution_campaign_spec_v41(
        development,
        campaign_id=campaign_id,
        policy=frozen_policy,
    )
    adapter = EventProcessEvolutionAdapterV41(
        development,
        frozen_policy,
        initial_family="exponential_hawkes",
    )
    return run_model_evolution_campaign_v41(output_root, spec, adapter)
