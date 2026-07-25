from __future__ import annotations

from datetime import datetime, timezone

from fma.hashing import sha256_value
from fma.v4.model_evolution_graph import (
    DevelopmentEvaluationV41,
    DevelopmentExecutionV41,
    EvolutionOperatorV41,
    EvolutionProposalV41,
    ModelCandidateV41,
    ModelEvolutionCampaignSpecV41,
    run_model_evolution_campaign_v41,
    verify_model_evolution_campaign_v41,
)


NOW = datetime(2026, 7, 23, tzinfo=timezone.utc)
DATA_HASH = sha256_value({"development": "fixture"})


def _candidate(
    family: str,
    generation: int,
    *,
    parent_hash: str | None = None,
    operator_hash: str | None = None,
) -> ModelCandidateV41:
    return ModelCandidateV41.seal(
        candidate_id=f"{family}.g{generation}",
        generation=generation,
        family=family,
        model_spec={"family": family},
        parent_candidate_hashes=[parent_hash] if parent_hash else [],
        operator_hashes=[operator_hash] if operator_hash else [],
        rationale=f"Evaluate the {family} family under frozen development gates.",
        expected_failure_modes=["development calibration may fail"],
    )


class _EvolutionFixture:
    def __init__(self) -> None:
        self.execute_calls = 0

    def initial_candidates(self, spec):
        return [_candidate("seed", 0)]

    def execute(self, spec, candidate):
        self.execute_calls += 1
        score = {"seed": 0.0, "repair": 2.0, "distractor": 0.5}[candidate.family]
        return DevelopmentExecutionV41.seal(
            candidate_hash=candidate.candidate_hash,
            development_data_hash=spec.development_data_hash,
            converged=True,
            metrics={"score": score},
            domain_payload={"family": candidate.family},
        )

    def evaluate(self, spec, candidate, execution):
        gates = {
            "calibrated": candidate.family == "repair",
            "predictive": candidate.family != "seed",
        }
        diagnostics = []
        if not gates["calibrated"]:
            diagnostics.append("calibration_failure")
        if not gates["predictive"]:
            diagnostics.append("predictive_failure")
        return DevelopmentEvaluationV41.seal(
            candidate_hash=candidate.candidate_hash,
            execution_hash=execution.execution_hash,
            evaluator_epoch=spec.evaluator_epoch,
            gates=gates,
            metrics={"score": execution.metrics["score"]},
            utility=execution.metrics["score"],
            diagnostic_codes=sorted(diagnostics),
            disposition="advance" if all(gates.values()) else "mutate",
        )

    def evolve(
        self,
        spec,
        candidate,
        evaluation,
        failure,
        next_generation,
    ):
        proposals = []
        for index, family in enumerate(("repair", "distractor")):
            operator = EvolutionOperatorV41.seal(
                operator_id=f"replace.{candidate.family}.{family}.g{next_generation}",
                kind="replace_skeleton",
                source_candidate_hash=candidate.candidate_hash,
                source_evaluation_hash=evaluation.evaluation_hash,
                failure_signature_hash=failure.failure_hash,
                diagnostic_codes=failure.diagnostic_codes,
                target_family=family,
                rationale=f"Replace {candidate.family} with {family} after verifier failure.",
            )
            child = _candidate(
                family,
                next_generation,
                parent_hash=candidate.candidate_hash,
                operator_hash=operator.operator_hash,
            )
            proposals.append(
                EvolutionProposalV41(
                    operator=operator,
                    candidate=child,
                    priority=1.0 - index * 0.1,
                )
            )
        return proposals


def _spec() -> ModelEvolutionCampaignSpecV41:
    return ModelEvolutionCampaignSpecV41.seal(
        campaign_id="model_evolution_fixture_v41",
        evaluator_epoch="development_fixture_epoch",
        objective="Evolve a failed seed into a calibrated development champion.",
        development_data_hash=DATA_HASH,
        required_gates=["calibrated", "predictive"],
        max_generations=2,
        beam_width=2,
        max_candidates=3,
        created_at=NOW,
    )


def test_dynamic_graph_evolves_failed_candidate_and_freezes_unqualified_champion(
    tmp_path,
) -> None:
    adapter = _EvolutionFixture()
    spec = _spec()
    outcome = run_model_evolution_campaign_v41(tmp_path, spec, adapter)

    assert adapter.execute_calls == 3
    assert outcome.report.generation_count == 2
    assert {candidate.family for candidate in outcome.candidates} == {
        "seed",
        "repair",
        "distractor",
    }
    assert len(outcome.operators) == 2
    assert outcome.champion is not None
    champion = next(
        candidate
        for candidate in outcome.candidates
        if candidate.candidate_hash == outcome.champion.candidate_hash
    )
    assert champion.family == "repair"
    assert outcome.report.terminal_status == "development_champion_frozen"
    assert outcome.report.qualification_granted is False
    assert outcome.report.private_confirmation_consumed is False
    assert not outcome.graph.project_state().promotions
    assert verify_model_evolution_campaign_v41(outcome, spec)

    relations = {edge.relation for edge in outcome.graph.project_state().edges}
    assert "derived_from" in relations
    assert "learned_from_failure" in relations
    assert "evaluated_by" in relations

    before = outcome.graph.store.event_path.read_bytes()

    class _NoCalls(_EvolutionFixture):
        def initial_candidates(self, spec):
            raise AssertionError("completed campaign regenerated candidates")

    resumed = run_model_evolution_campaign_v41(tmp_path, spec, _NoCalls())
    assert resumed.report == outcome.report
    assert resumed.graph.store.event_path.read_bytes() == before
