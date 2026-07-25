from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

import pytest

from fma.v4.codex_open_evolution import (
    FixtureOpenEvolutionTransportV42,
    GeneratedModelDraftV42,
    OpenEvolutionGenerationResponseV42,
)
from fma.v4.event_process_evolution import EventProcessEvolutionPolicyV41
from fma.v4.open_evolution_kernel import (
    ModelSymbolV42,
    OpenModelCandidateV42,
    PrimitiveApplicationV42,
    run_open_evolution_campaign_v42,
    verify_open_evolution_campaign_v42,
)
from fma.v4.recursive_event_process_evolution import (
    EVENT_PROCESS_RECURSIVE_ADAPTER_ID_V43,
    RecursiveEventProcessEvolutionAdapterV43,
    compile_recursive_event_process_v43,
    event_process_recursive_grammar_v43,
    event_process_topology_registry_v43,
    recursive_event_process_campaign_spec_v43,
)
from fma.v4.unseen_event_process import (
    EarthquakeEventV40,
    USGSCatalogQueryV40,
    USGSCatalogSnapshotV40,
)


NOW = datetime(2026, 7, 24, tzinfo=timezone.utc)


def _development_snapshot() -> USGSCatalogSnapshotV40:
    query = USGSCatalogQueryV40.seal(
        phase="development",
        start=datetime(2023, 1, 1, tzinfo=timezone.utc),
        end_exclusive=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    events = []
    for index in range(56):
        origin = query.start + timedelta(
            days=(index + 1) * 365.0 / 57.0,
            minutes=(index * 19) % 240,
        )
        events.append(
            EarthquakeEventV40(
                event_id=f"ci_recursive_{index:03d}",
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
            b"synthetic-recursive-development-catalog"
        ).hexdigest(),
        events=events,
        retrieved_at=NOW,
    )


def _mixture_draft(component_count: int, *, family: str | None = None):
    symbols = [
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
            unit="event_history",
        ),
        ModelSymbolV42(
            symbol_id="intensity",
            role="output",
            unit="rate_per_day",
        ),
    ]
    applications = [
        PrimitiveApplicationV42(
            application_id="background_component",
            primitive_id="constant_background",
            inputs=["background"],
            output="background_rate",
        )
    ]
    for index in range(component_count):
        symbols.extend(
            [
                ModelSymbolV42(
                    symbol_id=f"branching_{index}",
                    role="parameter",
                    unit="unitless",
                ),
                ModelSymbolV42(
                    symbol_id=f"decay_{index}",
                    role="parameter",
                    unit="rate_per_day",
                ),
                ModelSymbolV42(
                    symbol_id=f"memory_rate_{index}",
                    role="derived",
                    unit="rate_per_day",
                ),
            ]
        )
        applications.append(
            PrimitiveApplicationV42(
                application_id=f"memory_component_{index}",
                primitive_id="exponential_memory",
                inputs=[
                    "event_history",
                    f"branching_{index}",
                    f"decay_{index}",
                ],
                output=f"memory_rate_{index}",
            )
        )
        if index < component_count - 1:
            symbols.append(
                ModelSymbolV42(
                    symbol_id=f"partial_rate_{index}",
                    role="derived",
                    unit="rate_per_day",
                )
            )
    left = "background_rate"
    for index in range(component_count):
        output = (
            "intensity"
            if index == component_count - 1
            else f"partial_rate_{index}"
        )
        applications.append(
            PrimitiveApplicationV42(
                application_id=f"add_component_{index}",
                primitive_id="add_rate",
                inputs=[left, f"memory_rate_{index}"],
                output=output,
            )
        )
        left = output
    return GeneratedModelDraftV42(
        family=family or f"model_named_mixture_k{component_count}",
        kind="add_mechanism",
        symbols=symbols,
        applications=applications,
        assumptions=[
            "All exponential components share one observed event history.",
            "Each memory component has distinct branching and decay parameters.",
        ],
        transformation_summary=(
            f"Use {component_count} exponential memory components in one "
            "connected additive intensity tree."
        ),
        rationale=(
            "Increase temporal resolution by one component after a frozen "
            "parameter-boundary diagnostic."
        ),
        expected_failure_modes=[
            "A component may be weakly identifiable.",
            "A decay rate may remain on an optimization boundary.",
        ],
        priority=1.0,
    )


def _candidate(component_count: int, *, family: str):
    draft = _mixture_draft(component_count, family=family)
    return OpenModelCandidateV42.seal(
        candidate_id=f"compiler.fixture.k{component_count}",
        generation=0,
        family=family,
        source="seed",
        proposed_by="model",
        executable_adapter_id=EVENT_PROCESS_RECURSIVE_ADAPTER_ID_V43,
        symbols=draft.symbols,
        applications=draft.applications,
        assumptions=draft.assumptions,
        rationale=draft.rationale,
        expected_failure_modes=draft.expected_failure_modes,
    )


@pytest.mark.parametrize("component_count", [1, 2, 4])
def test_registry_compiles_variable_k_independent_of_family_label(
    component_count,
) -> None:
    candidate = _candidate(
        component_count,
        family=f"arbitrary_family_label_{component_count}",
    )
    compiled = compile_recursive_event_process_v43(candidate)
    assert compiled.executor_key == "exponential_mixture_hawkes"
    assert compiled.component_count == component_count
    assert compiled.primitive_counts["add_rate"] == component_count
    assert compiled.primitive_counts["exponential_memory"] == component_count


def test_registry_rejects_component_count_outside_frozen_range() -> None:
    with pytest.raises(ValueError, match="matched 0"):
        compile_recursive_event_process_v43(
            _candidate(5, family="five_component_candidate")
        )


def test_registry_rejects_shared_component_parameters() -> None:
    draft = _mixture_draft(2, family="shared_parameter_candidate")
    applications = list(draft.applications)
    memory_index = next(
        index
        for index, item in enumerate(applications)
        if item.application_id == "memory_component_1"
    )
    applications[memory_index] = PrimitiveApplicationV42(
        application_id="memory_component_1",
        primitive_id="exponential_memory",
        inputs=["event_history", "branching_0", "decay_0"],
        output="memory_rate_1",
    )
    candidate = OpenModelCandidateV42.seal(
        candidate_id="compiler.fixture.shared_parameters",
        generation=0,
        family=draft.family,
        source="seed",
        proposed_by="model",
        executable_adapter_id=EVENT_PROCESS_RECURSIVE_ADAPTER_ID_V43,
        symbols=draft.symbols,
        applications=applications,
        assumptions=draft.assumptions,
        rationale=draft.rationale,
        expected_failure_modes=draft.expected_failure_modes,
    )
    with pytest.raises(ValueError, match="distinct parameter symbols"):
        compile_recursive_event_process_v43(candidate)


def test_registry_rejects_disconnected_application() -> None:
    draft = _mixture_draft(2, family="disconnected_candidate")
    symbols = list(draft.symbols)
    symbols.extend(
        [
            ModelSymbolV42(
                symbol_id="unused_background",
                role="parameter",
                unit="rate_per_day",
            ),
            ModelSymbolV42(
                symbol_id="unused_rate",
                role="derived",
                unit="rate_per_day",
            ),
        ]
    )
    applications = [
        *draft.applications,
        PrimitiveApplicationV42(
            application_id="unused_background_component",
            primitive_id="constant_background",
            inputs=["unused_background"],
            output="unused_rate",
        ),
    ]
    candidate = OpenModelCandidateV42.seal(
        candidate_id="compiler.fixture.disconnected",
        generation=0,
        family=draft.family,
        source="seed",
        proposed_by="model",
        executable_adapter_id=EVENT_PROCESS_RECURSIVE_ADAPTER_ID_V43,
        symbols=symbols,
        applications=applications,
        assumptions=draft.assumptions,
        rationale=draft.rationale,
        expected_failure_modes=draft.expected_failure_modes,
    )
    with pytest.raises(ValueError, match="unused applications"):
        compile_recursive_event_process_v43(candidate)


def test_recursive_loop_calls_generator_again_from_generated_failure(
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
        # Every mixture decay is bounded below by 1/60, so this forces each
        # generated mixture to produce another graph-native failure.
        maximum_interior_hawkes_decay_rate_per_day=0.01,
    )

    def propose(request):
        current_count = sum(
            item.primitive_id == "exponential_memory"
            for item in request.current_candidate.applications
        )
        target_count = current_count + 1
        assert target_count in {2, 3}
        assert "parameter_boundary" in request.failure.diagnostic_codes
        return OpenEvolutionGenerationResponseV42(
            request_hash=request.request_hash,
            action="propose",
            proposals=[_mixture_draft(target_count)],
            rationale=(
                "Add exactly one memory component after the prior boundary failure."
            ),
        )

    transport = FixtureOpenEvolutionTransportV42(
        propose, generator_id="recursive_fixture_generator_v43"
    )
    grammar = event_process_recursive_grammar_v43()
    registry = event_process_topology_registry_v43()
    spec = recursive_event_process_campaign_spec_v43(
        development,
        grammar,
        registry,
        campaign_id="recursive_event_process_fixture_v43",
        policy=policy,
    )
    adapter = RecursiveEventProcessEvolutionAdapterV43(
        development, transport, policy, registry
    )
    outcome = run_open_evolution_campaign_v42(
        tmp_path, spec, grammar, adapter
    )

    assert len(transport.requests) == 2
    assert len(outcome.generation_calls) == 2
    assert len(
        [
            node
            for node in outcome.graph.project_state().nodes
            if node.node_kind == "generation_call"
        ]
    ) == 2
    component_counts = sorted(
        compile_recursive_event_process_v43(
            candidate, registry
        ).component_count
        for candidate in outcome.candidates
        if candidate.source in {"seed", "generated"}
    )
    assert component_counts == [1, 2, 3]
    generated = sorted(
        (
            candidate
            for candidate in outcome.candidates
            if candidate.source == "generated"
        ),
        key=lambda item: item.generation,
    )
    assert generated[1].parent_candidate_hashes == [
        generated[0].candidate_hash
    ]
    assert any(
        item.family == "weibull_renewal"
        and item.source == "prescribed"
        for item in outcome.candidates
    )
    assert outcome.report.terminal_status in {
        "development_champion_frozen",
        "no_development_candidate",
    }
    assert outcome.report.generated_operator_count == 2
    assert outcome.report.prescribed_operator_count == 1
    assert outcome.report.private_confirmation_consumed is False
    assert outcome.report.qualification_granted is False
    assert verify_open_evolution_campaign_v42(outcome, spec, grammar)
