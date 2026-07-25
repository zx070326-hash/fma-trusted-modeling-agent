from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

from fma.v4.codex_open_evolution import (
    FixtureOpenEvolutionTransportV42,
    GeneratedModelDraftV42,
    OpenEvolutionGenerationResponseV42,
)
from fma.v4.event_process_evolution import EventProcessEvolutionPolicyV41
from fma.v4.event_process_open_evolution import (
    EventProcessOpenEvolutionAdapterV42,
    compile_event_process_candidate_v42,
    event_process_open_campaign_spec_v42,
    event_process_open_grammar_v42,
)
from fma.v4.open_evolution_kernel import (
    ModelSymbolV42,
    PrimitiveApplicationV42,
    run_open_evolution_campaign_v42,
    verify_open_evolution_campaign_v42,
)
from fma.v4.unseen_event_process import (
    EarthquakeEventV40,
    USGSCatalogQueryV40,
    USGSCatalogSnapshotV40,
)


NOW = datetime(2026, 7, 23, tzinfo=timezone.utc)


def _development_snapshot() -> USGSCatalogSnapshotV40:
    query = USGSCatalogQueryV40.seal(
        phase="development",
        start=datetime(2023, 1, 1, tzinfo=timezone.utc),
        end_exclusive=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    events = []
    for index in range(70):
        origin = query.start + timedelta(
            days=(index + 1) * 365.0 / 71.0,
            minutes=(index * 17) % 180,
        )
        events.append(
            EarthquakeEventV40(
                event_id=f"ci_open_evolution_{index:03d}",
                origin_time=origin,
                magnitude=2.5,
                latitude=34.0,
                longitude=-118.0,
                depth_km=8.0,
            )
        )
    return USGSCatalogSnapshotV40.seal(
        query=query,
        response_sha256=hashlib.sha256(
            b"synthetic-open-development-only-catalog"
        ).hexdigest(),
        events=events,
        retrieved_at=NOW,
    )


def _two_timescale_draft(*, invalid_history_unit: bool = False):
    suffix = "invalid" if invalid_history_unit else "valid"
    return GeneratedModelDraftV42(
        family=(
            "invalid_dimensional_memory"
            if invalid_history_unit
            else "two_timescale_triggering"
        ),
        kind="invent_structure",
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
                symbol_id="event_history",
                role="observed",
                unit="event_count" if invalid_history_unit else "event_history",
            ),
            ModelSymbolV42(
                symbol_id="fast_branching",
                role="parameter",
                unit="unitless",
            ),
            ModelSymbolV42(
                symbol_id="fast_decay",
                role="parameter",
                unit="rate_per_day",
            ),
            ModelSymbolV42(
                symbol_id="fast_rate",
                role="derived",
                unit="rate_per_day",
            ),
            ModelSymbolV42(
                symbol_id="slow_branching",
                role="parameter",
                unit="unitless",
            ),
            ModelSymbolV42(
                symbol_id="slow_decay",
                role="parameter",
                unit="rate_per_day",
            ),
            ModelSymbolV42(
                symbol_id="slow_rate",
                role="derived",
                unit="rate_per_day",
            ),
            ModelSymbolV42(
                symbol_id="partial_rate",
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
                application_id=f"background_{suffix}",
                primitive_id="constant_background",
                inputs=["background"],
                output="background_rate",
            ),
            PrimitiveApplicationV42(
                application_id=f"fast_memory_{suffix}",
                primitive_id="exponential_memory",
                inputs=["event_history", "fast_branching", "fast_decay"],
                output="fast_rate",
            ),
            PrimitiveApplicationV42(
                application_id=f"slow_memory_{suffix}",
                primitive_id="exponential_memory",
                inputs=["event_history", "slow_branching", "slow_decay"],
                output="slow_rate",
            ),
            PrimitiveApplicationV42(
                application_id=f"partial_sum_{suffix}",
                primitive_id="add_rate",
                inputs=["background_rate", "fast_rate"],
                output="partial_rate",
            ),
            PrimitiveApplicationV42(
                application_id=f"intensity_sum_{suffix}",
                primitive_id="add_rate",
                inputs=["partial_rate", "slow_rate"],
                output="intensity",
            ),
        ],
        assumptions=[
            "Fast and slow triggering components share one stationary event history.",
            "The two exponential decay rates describe distinct temporal scales.",
        ],
        transformation_summary=(
            "Split one exponential memory response into fast and slow components."
        ),
        rationale=(
            "A two-timescale intensity can represent immediate aftershocks and "
            "persistent triggering that a single exponential kernel misses."
        ),
        expected_failure_modes=[
            "One component may collapse to negligible branching mass.",
            "The two decay rates may not be separately identifiable.",
        ],
        priority=1.0 if not invalid_history_unit else 0.8,
    )


def test_generated_structure_is_compiled_fitted_and_audited_in_one_graph(
    tmp_path,
) -> None:
    development = _development_snapshot()
    policy = EventProcessEvolutionPolicyV41.seal(
        split_fraction=0.7,
        minimum_events_per_slice=5,
        minimum_validation_log_score_lift_nat_per_event=0.0,
        minimum_time_rescaling_ks_pvalue=0.01,
        maximum_compensator_count_relative_error=0.9,
        maximum_hawkes_branching_ratio=0.95,
        maximum_interior_hawkes_decay_rate_per_day=0.01,
    )

    def propose(request):
        serialized = request.model_dump_json()
        assert request.private_evidence_exposed is False
        assert request.authority_fields_exposed is False
        assert request.tools_permitted is False
        assert "confirmation_snapshot" not in serialized
        assert "2024-01-01" not in serialized
        assert "qualification_granted" not in serialized
        return OpenEvolutionGenerationResponseV42(
            request_hash=request.request_hash,
            action="propose",
            proposals=[
                _two_timescale_draft(),
                _two_timescale_draft(invalid_history_unit=True),
            ],
            rationale="exercise one executable and one quarantined generated topology",
        )

    transport = FixtureOpenEvolutionTransportV42(propose)
    grammar = event_process_open_grammar_v42()
    spec = event_process_open_campaign_spec_v42(
        development,
        grammar,
        campaign_id="event_process_open_fixture_v42",
        policy=policy,
    )
    adapter = EventProcessOpenEvolutionAdapterV42(
        development, transport, policy
    )
    outcome = run_open_evolution_campaign_v42(
        tmp_path, spec, grammar, adapter
    )

    generated = [
        item for item in outcome.candidates if item.source == "generated"
    ]
    assert len(generated) == 2
    valid = next(
        item for item in generated if item.family == "two_timescale_triggering"
    )
    invalid = next(
        item for item in generated if item.family == "invalid_dimensional_memory"
    )
    assert compile_event_process_candidate_v42(valid).structure == (
        "two_timescale_hawkes"
    )
    valid_execution = next(
        item for item in outcome.executions
        if item.candidate_hash == valid.candidate_hash
    )
    assert valid_execution.domain_payload["compiled"]["structure"] == (
        "two_timescale_hawkes"
    )
    invalid_validation = next(
        item for item in outcome.validations
        if item.candidate_hash == invalid.candidate_hash
    )
    assert invalid_validation.admitted is False
    assert invalid_validation.checks["units_valid"] is False
    assert all(
        item.candidate_hash != invalid.candidate_hash
        for item in outcome.executions
    )
    assert len(outcome.generation_calls) == 1
    assert outcome.generation_calls[0].transport == "fixture"
    assert len(
        [
            node
            for node in outcome.graph.project_state().nodes
            if node.node_kind == "generation_call"
        ]
    ) == 1
    assert outcome.report.private_confirmation_consumed is False
    assert outcome.report.qualification_granted is False
    assert verify_open_evolution_campaign_v42(outcome, spec, grammar)
