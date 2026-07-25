from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fma.v4 import (
    default_product_vertical_slice_spec_v40,
    run_product_vertical_slice_v40,
    verify_product_vertical_slice_v40,
)


ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 7, 23, tzinfo=timezone.utc)


def test_v311_to_v313_vertical_slice_is_resumable_and_atomic(tmp_path) -> None:
    spec = default_product_vertical_slice_spec_v40(
        ROOT,
        created_at=NOW,
        slice_id="v4_vertical_slice_test",
    )
    outcome = run_product_vertical_slice_v40(tmp_path, spec)
    assert outcome.report.source_verifications == {
        "verify_v311": True,
        "verify_v312": True,
        "verify_v313_development": True,
        "verify_v313_confirmation": True,
    }
    assert outcome.report.atomic_decision == "rejected"
    assert outcome.report.terminal_status == "scientific_concepts_rejected_v40"
    assert not outcome.report.active_concept_versions
    assert not outcome.report.real_world_execution_permitted
    epistemic = outcome.epistemic_graph.project_state()
    assert len(epistemic.nodes) == 1
    assert epistemic.nodes[0].node_kind == "failure_signature"
    assert verify_product_vertical_slice_v40(outcome, spec)

    resumed = run_product_vertical_slice_v40(tmp_path, spec)
    assert resumed.report == outcome.report
    assert resumed.graph.project_state().snapshot == outcome.graph.project_state().snapshot
    assert verify_product_vertical_slice_v40(resumed, spec)
