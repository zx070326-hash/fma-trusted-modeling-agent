from __future__ import annotations

from datetime import datetime, timezone

import pytest

from fma.hashing import sha256_value
from fma.v4.open_evolution_kernel import (
    DevelopmentEvaluationV42,
    DevelopmentExecutionV42,
    ExecutionAttemptV42,
    GenerationCallEvidenceV42,
    HybridEvolutionOperatorV42,
    OpenEvolutionCampaignSpecV42,
    OpenEvolutionProposalV42,
    OpenFailureSignatureV42,
    OpenModelCandidateV42,
    OpenModelGrammarV42,
    PrimitiveApplicationV42,
    PrimitiveRuleV42,
    ModelSymbolV42,
    run_open_evolution_campaign_v42,
    verify_open_evolution_campaign_v42,
)


NOW = datetime(2026, 7, 23, tzinfo=timezone.utc)
DATA_HASH = sha256_value({"development": "open-evolution-fixture"})
ADAPTER_ID = "fixture_local_v42"
ADAPTER_CONTRACT_HASH = sha256_value(
    {
        "adapter": ADAPTER_ID,
        "execution": "deterministic_score_fixture_v1",
        "evaluation": "two_gate_fixture_v1",
    }
)


def _generation_evidence(tag: str) -> GenerationCallEvidenceV42:
    request_payload = {
        "fixture_request": tag,
        "private_evidence_exposed": False,
        "authority_fields_exposed": False,
        "tools_permitted": False,
    }
    request_hash = sha256_value(request_payload)
    response_payload = {
        "fixture_response": tag,
        "request_hash": request_hash,
    }
    return GenerationCallEvidenceV42.seal(
        generator_id="fixture_generator_v42",
        transport="fixture",
        request_hash=request_hash,
        response_hash=sha256_value(response_payload),
        request_payload=request_payload,
        response_payload=response_payload,
    )


def _grammar() -> OpenModelGrammarV42:
    return OpenModelGrammarV42.seal(
        grammar_id="typed_rate_grammar_v42",
        seed_families=["constant_rate_seed"],
        primitives=[
            PrimitiveRuleV42(
                primitive_id="add_rate",
                role="combiner",
                input_units=["rate_per_day", "rate_per_day"],
                output_unit="rate_per_day",
            ),
            PrimitiveRuleV42(
                primitive_id="history_response",
                role="mechanism",
                input_units=["event_count"],
                output_unit="rate_per_day",
            ),
            PrimitiveRuleV42(
                primitive_id="identity_rate",
                role="transform",
                input_units=["rate_per_day"],
                output_unit="rate_per_day",
            ),
            PrimitiveRuleV42(
                primitive_id="renewal_hazard",
                role="mechanism",
                input_units=["day", "unitless", "day"],
                output_unit="rate_per_day",
            ),
        ],
        executable_adapter_ids=[ADAPTER_ID],
        executable_adapter_hashes={
            ADAPTER_ID: ADAPTER_CONTRACT_HASH,
        },
        forbidden_tokens=["private_confirmation", "qualification_granted"],
        max_symbols=12,
        max_applications=8,
    )


def _spec(campaign_id: str, grammar: OpenModelGrammarV42, *, generations: int = 2):
    return OpenEvolutionCampaignSpecV42.seal(
        campaign_id=campaign_id,
        evaluator_epoch="open_evolution_fixture_epoch",
        objective=(
            "Expand a typed model space with prescribed and generated operators "
            "while preserving independent development verification."
        ),
        development_data_hash=DATA_HASH,
        grammar_hash=grammar.grammar_hash,
        required_development_gates=["calibrated", "predictive"],
        max_generations=generations,
        max_candidates=4,
        prescribed_quota_per_failure=1,
        generated_quota_per_failure=2,
        max_recovery_attempts=3,
        created_at=NOW,
    )


def _seed() -> OpenModelCandidateV42:
    return OpenModelCandidateV42.seal(
        candidate_id="constant_rate_seed.g0",
        generation=0,
        family="constant_rate_seed",
        source="seed",
        proposed_by="model",
        executable_adapter_id=ADAPTER_ID,
        symbols=[
            ModelSymbolV42(
                symbol_id="baseline",
                role="parameter",
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
                application_id="constant.intensity",
                primitive_id="identity_rate",
                inputs=["baseline"],
                output="intensity",
            )
        ],
        assumptions=["Event intensity is constant over the development window."],
        rationale="Use a minimal constant-rate seed before failure-driven expansion.",
        expected_failure_modes=["Temporal dependence may violate predictive gates."],
    )


def _renewal_candidate(
    parent: OpenModelCandidateV42,
    operator: HybridEvolutionOperatorV42,
) -> OpenModelCandidateV42:
    return OpenModelCandidateV42.seal(
        candidate_id="renewal_hazard.g1",
        generation=1,
        family="renewal_hazard",
        source="prescribed",
        proposed_by="harness",
        executable_adapter_id=ADAPTER_ID,
        symbols=[
            ModelSymbolV42(symbol_id="age", role="observed", unit="day"),
            ModelSymbolV42(symbol_id="shape", role="parameter", unit="unitless"),
            ModelSymbolV42(symbol_id="scale", role="parameter", unit="day"),
            ModelSymbolV42(
                symbol_id="intensity", role="output", unit="rate_per_day"
            ),
        ],
        applications=[
            PrimitiveApplicationV42(
                application_id="renewal.intensity",
                primitive_id="renewal_hazard",
                inputs=["age", "shape", "scale"],
                output="intensity",
            )
        ],
        assumptions=["Interarrival age is sufficient for the renewal hazard."],
        parent_candidate_hashes=[parent.candidate_hash],
        operator_hashes=[operator.operator_hash],
        rationale="Apply the prescribed renewal repair after the constant seed fails.",
        expected_failure_modes=["Renewal calibration may still fail."],
    )


def _history_candidate(
    parent: OpenModelCandidateV42,
    operator: HybridEvolutionOperatorV42,
    *,
    invalid_unit: bool,
) -> OpenModelCandidateV42:
    suffix = "invalid" if invalid_unit else "valid"
    return OpenModelCandidateV42.seal(
        candidate_id=f"history_augmented_rate.{suffix}.g1",
        generation=1,
        family=(
            "dimensionally_invalid_history"
            if invalid_unit
            else "history_augmented_rate"
        ),
        source="generated",
        proposed_by="model",
        executable_adapter_id=ADAPTER_ID,
        symbols=[
            ModelSymbolV42(
                symbol_id="baseline", role="parameter", unit="rate_per_day"
            ),
            ModelSymbolV42(
                symbol_id="history",
                role="observed",
                unit="day" if invalid_unit else "event_count",
            ),
            ModelSymbolV42(
                symbol_id="base_rate", role="derived", unit="rate_per_day"
            ),
            ModelSymbolV42(
                symbol_id="memory_rate", role="derived", unit="rate_per_day"
            ),
            ModelSymbolV42(
                symbol_id="intensity", role="output", unit="rate_per_day"
            ),
        ],
        applications=[
            PrimitiveApplicationV42(
                application_id="history.base",
                primitive_id="identity_rate",
                inputs=["baseline"],
                output="base_rate",
            ),
            PrimitiveApplicationV42(
                application_id="history.memory",
                primitive_id="history_response",
                inputs=["history"],
                output="memory_rate",
            ),
            PrimitiveApplicationV42(
                application_id="history.total",
                primitive_id="add_rate",
                inputs=["base_rate", "memory_rate"],
                output="intensity",
            ),
        ],
        assumptions=["Recent event history can alter the current event rate."],
        parent_candidate_hashes=[parent.candidate_hash],
        operator_hashes=[operator.operator_hash],
        rationale=(
            "Generate a history-sensitive structure not present in the seed family."
        ),
        expected_failure_modes=["History response may be miscalibrated."],
    )


class _OpenEvolutionFixture:
    adapter_id = ADAPTER_ID
    adapter_contract_hash = ADAPTER_CONTRACT_HASH

    def __init__(self, *, crash_once: bool = False, seed_passes: bool = False):
        self.crash_once = crash_once
        self.seed_passes = seed_passes
        self.execute_calls = 0
        self.idempotency_keys: list[str] = []

    def initial_candidates(self, spec, grammar):
        return [_seed()]

    def supports_candidate(self, candidate):
        return candidate.executable_adapter_id == self.adapter_id

    def execute(self, spec, candidate, attempt: ExecutionAttemptV42):
        self.execute_calls += 1
        self.idempotency_keys.append(attempt.idempotency_key)
        if self.crash_once:
            self.crash_once = False
            raise RuntimeError("synthetic interruption after durable execution attempt")
        score = {
            "constant_rate_seed": 0.0 if not self.seed_passes else 3.0,
            "renewal_hazard": 1.0,
            "history_augmented_rate": 2.0,
        }[candidate.family]
        return DevelopmentExecutionV42.seal(
            candidate_hash=candidate.candidate_hash,
            attempt_hash=attempt.attempt_hash,
            idempotency_key=attempt.idempotency_key,
            development_data_hash=spec.development_data_hash,
            converged=True,
            metrics={"score": score},
            domain_payload={"family": candidate.family},
        )

    def evaluate(self, spec, candidate, execution):
        passes = candidate.family != "constant_rate_seed" or self.seed_passes
        gates = {"calibrated": passes, "predictive": passes}
        return DevelopmentEvaluationV42.seal(
            candidate_hash=candidate.candidate_hash,
            execution_hash=execution.execution_hash,
            evaluator_epoch=spec.evaluator_epoch,
            gates=gates,
            metrics={"score": execution.metrics["score"]},
            utility=execution.metrics["score"],
            diagnostic_codes=[] if passes else ["missing_temporal_dependence"],
            disposition="advance" if passes else "mutate",
        )

    def prescribed_evolve(
        self, spec, grammar, candidate, evaluation, failure, next_generation
    ):
        operator = HybridEvolutionOperatorV42.seal(
            operator_id=f"prescribed.renewal.{candidate.candidate_hash[:8]}",
            channel="prescribed",
            kind="replace_skeleton",
            proposed_by="harness",
            source_candidate_hash=candidate.candidate_hash,
            source_evaluation_hash=evaluation.evaluation_hash,
            failure_signature_hash=failure.failure_hash,
            target_family="renewal_hazard",
            transformation_summary="Replace constant intensity with an age hazard.",
            rationale="Use a deterministic renewal repair for temporal dependence.",
        )
        return [
            OpenEvolutionProposalV42(
                operator=operator,
                candidate=_renewal_candidate(candidate, operator),
                priority=0.8,
            )
        ]

    def generated_evolve(
        self,
        spec,
        grammar,
        candidate,
        evaluation,
        failure: OpenFailureSignatureV42,
        next_generation,
    ):
        proposals = []
        for invalid, priority in ((False, 1.0), (True, 0.7)):
            family = (
                "dimensionally_invalid_history"
                if invalid
                else "history_augmented_rate"
            )
            operator = HybridEvolutionOperatorV42.seal(
                operator_id=(
                    f"generated.{family}.{candidate.candidate_hash[:8]}"
                ),
                channel="generated",
                kind="invent_structure",
                proposed_by="model",
                source_candidate_hash=candidate.candidate_hash,
                source_evaluation_hash=evaluation.evaluation_hash,
                failure_signature_hash=failure.failure_hash,
                target_family=family,
                transformation_summary=(
                    "Compose baseline, history response and additive rate primitives."
                ),
                rationale=(
                    "Generate a history-sensitive family after predictive failure."
                ),
            )
            proposals.append(
                OpenEvolutionProposalV42(
                    operator=operator,
                    candidate=_history_candidate(
                        candidate, operator, invalid_unit=invalid
                    ),
                    generation_evidence=_generation_evidence(family),
                    priority=priority,
                )
            )
        return proposals


def test_open_space_and_hybrid_evolution_share_one_graph_verifier(tmp_path):
    grammar = _grammar()
    spec = _spec("open_evolution_fixture_v42", grammar)
    adapter = _OpenEvolutionFixture()
    outcome = run_open_evolution_campaign_v42(
        tmp_path, spec, grammar, adapter
    )

    assert len(outcome.candidates) == 4
    assert {item.family for item in outcome.candidates} == {
        "constant_rate_seed",
        "renewal_hazard",
        "history_augmented_rate",
        "dimensionally_invalid_history",
    }
    assert "history_augmented_rate" not in grammar.seed_families
    invalid = next(
        item
        for item in outcome.candidates
        if item.family == "dimensionally_invalid_history"
    )
    invalid_validation = next(
        item
        for item in outcome.validations
        if item.candidate_hash == invalid.candidate_hash
    )
    assert invalid_validation.admitted is False
    assert invalid_validation.checks["units_valid"] is False
    assert invalid.candidate_hash not in {
        item.candidate_hash for item in outcome.executions
    }
    assert outcome.report.prescribed_operator_count == 1
    assert outcome.report.generated_operator_count == 2
    assert outcome.champion is not None
    champion = next(
        item
        for item in outcome.candidates
        if item.candidate_hash == outcome.champion.candidate_hash
    )
    assert champion.family == "history_augmented_rate"
    assert outcome.report.qualification_granted is False
    assert outcome.report.private_confirmation_consumed is False
    assert not outcome.graph.project_state().promotions
    node_kinds = {item.node_kind for item in outcome.graph.project_state().nodes}
    assert {
        "model_space",
        "model_proposal",
        "model_validation",
        "model_admission",
        "evolution_operator",
        "checkpoint",
    }.issubset(node_kinds)
    assert verify_open_evolution_campaign_v42(outcome, spec, grammar)


def test_incomplete_execution_recovers_with_graph_patch_and_same_idempotency_key(
    tmp_path,
):
    grammar = _grammar()
    spec = _spec(
        "open_evolution_recovery_fixture_v42",
        grammar,
        generations=1,
    )
    first = _OpenEvolutionFixture(crash_once=True, seed_passes=True)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        run_open_evolution_campaign_v42(tmp_path, spec, grammar, first)
    assert first.execute_calls == 1

    second = _OpenEvolutionFixture(seed_passes=True)
    outcome = run_open_evolution_campaign_v42(
        tmp_path, spec, grammar, second
    )
    assert second.execute_calls == 1
    assert first.idempotency_keys == second.idempotency_keys
    assert len(outcome.incidents) == 1
    assert len(outcome.recovery_patches) == 1
    assert any(item.phase == "reconciled" for item in outcome.checkpoints)
    state = outcome.graph.project_state()
    incident_node = next(
        item for item in state.nodes if item.node_kind == "incident"
    )
    patch_node = next(
        item for item in state.nodes if item.node_kind == "recovery_patch"
    )
    assert any(
        edge.source_node_hash == incident_node.node_hash
        and edge.target_node_hash == patch_node.node_hash
        and edge.relation == "learned_from_failure"
        for edge in state.edges
    )
    assert outcome.report.recovered_from_incomplete_run is True
    assert verify_open_evolution_campaign_v42(outcome, spec, grammar)

    before = outcome.graph.store.event_path.read_bytes()
    execute_calls = second.execute_calls
    resumed = run_open_evolution_campaign_v42(
        tmp_path, spec, grammar, second
    )
    assert resumed.report == outcome.report
    assert second.execute_calls == execute_calls
    assert resumed.graph.store.event_path.read_bytes() == before

    mismatched = _OpenEvolutionFixture(seed_passes=True)
    mismatched.adapter_contract_hash = sha256_value({"different": "adapter"})
    with pytest.raises(ValueError, match="differs from the frozen grammar"):
        run_open_evolution_campaign_v42(
            tmp_path, spec, grammar, mismatched
        )
    assert resumed.graph.store.event_path.read_bytes() == before


def test_committed_execution_is_reconciled_without_reexecuting_adapter(
    tmp_path, monkeypatch
):
    from fma.v4.graph_loop import GraphLoopStoreV40

    grammar = _grammar()
    spec = _spec(
        "open_evolution_committed_result_recovery_v42",
        grammar,
        generations=1,
    )
    first = _OpenEvolutionFixture(seed_passes=True)
    original = GraphLoopStoreV40.record_outcome
    interrupted = {"raised": False}

    def _interrupt_before_execution_outcome(self, node_hash, **kwargs):
        node = self.project_state().node_by_hash(node_hash)
        if node.node_kind == "execution" and not interrupted["raised"]:
            interrupted["raised"] = True
            raise RuntimeError("synthetic interruption after execution artifact commit")
        return original(self, node_hash, **kwargs)

    monkeypatch.setattr(
        GraphLoopStoreV40,
        "record_outcome",
        _interrupt_before_execution_outcome,
    )
    with pytest.raises(RuntimeError, match="after execution artifact commit"):
        run_open_evolution_campaign_v42(tmp_path, spec, grammar, first)
    assert first.execute_calls == 1

    monkeypatch.setattr(GraphLoopStoreV40, "record_outcome", original)
    second = _OpenEvolutionFixture(seed_passes=True)
    outcome = run_open_evolution_campaign_v42(
        tmp_path, spec, grammar, second
    )
    assert second.execute_calls == 0
    assert len(outcome.executions) == 1
    assert len(outcome.incidents) == 1
    assert outcome.report.recovered_from_incomplete_run is True
    assert verify_open_evolution_campaign_v42(outcome, spec, grammar)
