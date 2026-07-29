from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from fma.hashing import canonical_json
from fma.v6.positive_log_increment_v68 import (
    PositiveLogIncrementReplayAuthorityV68,
    PositiveLogIncrementThresholdsV68,
    PositiveScalarSeriesSnapshotV68,
    build_positive_log_increment_bundle_v68,
    compile_positive_log_increment_ir_v68,
    deterministic_positive_log_increment_hash_v68,
    execute_positive_log_increment_ir_v68,
    run_authenticated_positive_log_increment_replays_v68,
)
from fma.v6.positive_log_increment_verifier_v68 import (
    recompute_positive_log_increment_level_v68,
)


ROOT = Path(__file__).resolve().parents[1]


def _thresholds() -> PositiveLogIncrementThresholdsV68:
    return PositiveLogIncrementThresholdsV68.seal()


def _snapshot(
    task_id: str,
    values: np.ndarray,
) -> PositiveScalarSeriesSnapshotV68:
    return PositiveScalarSeriesSnapshotV68.seal(
        task_id=task_id,
        time_unit="year",
        state_unit="positive_index",
        times=np.arange(len(values), dtype=float).tolist(),
        observations=values.tolist(),
        source_id=f"{task_id}-source",
        fixture_only=True,
    )


def _growth_snapshot(
    *,
    task_id: str,
    mean: float,
    phi: float,
    sigma: float,
    seed: int,
    count: int = 72,
) -> PositiveScalarSeriesSnapshotV68:
    rng = np.random.default_rng(seed)
    growths = np.zeros(count - 1, dtype=float)
    growths[0] = mean
    for index in range(1, len(growths)):
        growths[index] = (
            mean
            + phi * (growths[index - 1] - mean)
            + rng.normal(0.0, sigma)
        )
    values = 100.0 * np.exp(np.concatenate(([0.0], np.cumsum(growths))))
    return _snapshot(task_id, values)


def test_v68_is_pure_and_does_not_import_legacy_model_bundles() -> None:
    source_path = ROOT / "fma" / "v6" / "positive_log_increment_v68.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert not any(
        module.startswith(("fma.v5_2", "fma.v5_6", "fma.v5_7"))
        for module in imported_modules
    )
    source = source_path.read_text(encoding="utf-8")
    assert "build_hybrid_ode_bundle" not in source
    assert "AdaptivePositiveSeriesBundle" not in source


def test_v68_drift_bundle_is_deterministic_and_l0_is_not_run() -> None:
    snapshot = _growth_snapshot(
        task_id="v68-log-drift",
        mean=0.04,
        phi=0.0,
        sigma=0.01,
        seed=102,
    )
    thresholds = _thresholds()
    first = build_positive_log_increment_bundle_v68(
        snapshot=snapshot,
        thresholds=thresholds,
    )
    second = build_positive_log_increment_bundle_v68(
        snapshot=snapshot,
        thresholds=thresholds,
    )
    assert first == second
    assert first.bundle_hash == second.bundle_hash
    assert first.model_ir_hash == compile_positive_log_increment_ir_v68(
        thresholds
    ).ir_hash
    assert deterministic_positive_log_increment_hash_v68(
        snapshot=snapshot,
        thresholds=thresholds,
    ) == deterministic_positive_log_increment_hash_v68(
        snapshot=snapshot,
        thresholds=thresholds,
    )
    assert [item.candidate_id for item in first.candidates] == [
        "log_growth_ar1",
        "log_random_walk_drift",
    ]
    assert first.selection_status == "SELECTED"
    assert first.selected_model_id == "log_random_walk_drift"
    assert [item.level for item in first.levels] == [
        "L0",
        "L1",
        "L2",
        "L3",
        "L4",
    ]
    assert [item.status for item in first.levels] == [
        "NOT_RUN",
        "PASS",
        "PASS",
        "PASS",
        "PASS",
    ]
    assert first.local_l0_l4_complete is False
    assert first.scientific_acceptance is False


def test_v68_typed_ir_execution_and_level_verification_are_hash_bound() -> None:
    snapshot = _growth_snapshot(
        task_id="v68-typed-ir",
        mean=0.04,
        phi=0.0,
        sigma=0.01,
        seed=109,
    )
    thresholds = _thresholds()
    model_ir = compile_positive_log_increment_ir_v68(thresholds)
    bundle = execute_positive_log_increment_ir_v68(
        model_ir=model_ir,
        snapshot=snapshot,
        thresholds=thresholds,
    )

    model_ir.assert_sealed()
    bundle.assert_sealed()
    assert bundle.model_ir_hash == model_ir.ir_hash
    assert recompute_positive_log_increment_level_v68(
        bundle=bundle,
        level="L2",
        model_ir=model_ir,
        snapshot=snapshot,
        thresholds=thresholds,
    ) == next(item for item in bundle.levels if item.level == "L2")

    tampered_ir = model_ir.model_copy(update={"threshold_hash": "f" * 64})
    with pytest.raises(ValueError, match="IR is not sealed"):
        execute_positive_log_increment_ir_v68(
            model_ir=tampered_ir,
            snapshot=snapshot,
            thresholds=thresholds,
        )

    tampered_bundle = bundle.model_copy(update={"model_ir_hash": "e" * 64})
    with pytest.raises(ValueError, match="bundle"):
        recompute_positive_log_increment_level_v68(
            bundle=tampered_bundle,
            level="L2",
            model_ir=model_ir,
            snapshot=snapshot,
            thresholds=thresholds,
        )

    original_l0 = next(item for item in bundle.levels if item.level == "L0")
    fake_l0_payload = original_l0.model_dump(
        mode="json",
        exclude={"status", "checks", "evidence_hash"},
    )
    fake_l0 = type(original_l0).seal(
        **fake_l0_payload,
        status="PASS",
        checks={key: True for key in original_l0.checks},
    )
    fake_bundle_payload = bundle.model_dump(
        mode="json",
        exclude={"levels", "local_l0_l4_complete", "bundle_hash"},
    )
    fake_levels = [
        fake_l0 if item.level == "L0" else item
        for item in bundle.levels
    ]
    fake_bundle = type(bundle).seal(
        **fake_bundle_payload,
        levels=fake_levels,
        local_l0_l4_complete=all(
            item.status == "PASS" for item in fake_levels
        ),
    )
    with pytest.raises(ValueError, match="input-bound recomputation"):
        recompute_positive_log_increment_level_v68(
            bundle=fake_bundle,
            level="L0",
            model_ir=model_ir,
            snapshot=snapshot,
            thresholds=thresholds,
        )


def test_v68_ar1_is_selected_only_when_materially_better() -> None:
    bundle = build_positive_log_increment_bundle_v68(
        snapshot=_growth_snapshot(
            task_id="v68-log-ar1",
            mean=0.04,
            phi=0.85,
            sigma=0.02,
            seed=103,
        ),
        thresholds=_thresholds(),
    )
    assert bundle.selection_status == "SELECTED"
    assert bundle.selected_model_id == "log_growth_ar1"
    selected = next(
        item
        for item in bundle.candidates
        if item.candidate_id == bundle.selected_model_id
    )
    assert selected.local_development_admissible is True
    assert selected.same_family_ar1_relative_improvement is not None
    assert (
        selected.same_family_ar1_relative_improvement
        >= _thresholds().minimum_growth_ar1_validation_relative_improvement
    )


def test_v68_local_fresh_replays_cannot_close_l0(tmp_path: Path) -> None:
    snapshot = _growth_snapshot(
        task_id="v68-replay",
        mean=0.04,
        phi=0.0,
        sigma=0.01,
        seed=102,
    )
    thresholds = _thresholds()
    replay_input = tmp_path / "positive-log-increment-replay.json"
    replay_input.write_text(
        canonical_json(
            {
                "snapshot": snapshot.model_dump(mode="json"),
                "thresholds": thresholds.model_dump(mode="json"),
            }
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    authority = PositiveLogIncrementReplayAuthorityV68(
        key_id="v68-test-replay",
        secret=b"r" * 32,
    )
    assert not hasattr(authority, "issue")
    receipts = run_authenticated_positive_log_increment_replays_v68(
        replay_input,
        authority=authority,
    )
    bundle = build_positive_log_increment_bundle_v68(
        snapshot=snapshot,
        thresholds=thresholds,
        replay_receipts=receipts,
        replay_authority=authority,
    )
    assert [item.replay_index for item in receipts] == [1, 2]
    assert len({item.process_id for item in receipts}) == 2
    assert len({item.deterministic_output_hash for item in receipts}) == 1
    assert all(authority.verify(item) for item in receipts)
    assert [item.status for item in bundle.levels] == [
        "NOT_RUN",
        "PASS",
        "PASS",
        "PASS",
        "PASS",
    ]
    assert bundle.local_l0_l4_complete is False
    assert bundle.scientific_acceptance is False
    assert bundle.fixture_only is True
    assert bundle.interval_claim_ceiling == "diagnostic_interval_quality_only"
    assert bundle.temporal_dependence_coverage_guaranteed is False
    assert bundle.finite_sample_coverage_guaranteed is False
    assert bundle.post_selection_coverage_guaranteed is False
    assert bundle.replay_receipts_are_external_qualification_evidence is False
    assert bundle.causal_mechanism_identified is False
    assert bundle.real_world_capability_established is False
    assert bundle.scientific_qualification_granted is False
    assert bundle.real_world_action_authorized is False


def test_v68_short_and_nonpositive_inputs_are_rejected() -> None:
    with pytest.raises(ValidationError):
        _snapshot("v68-short", np.full(33, 100.0))
    with pytest.raises(ValidationError, match="must be positive"):
        _snapshot(
            "v68-nonpositive",
            np.concatenate((np.full(33, 100.0), [0.0])),
        )


def test_v68_perfect_persistence_baseline_forces_abstention() -> None:
    bundle = build_positive_log_increment_bundle_v68(
        snapshot=_snapshot("v68-no-signal", np.full(40, 100.0)),
        thresholds=_thresholds(),
    )
    assert bundle.selection_status == "ABSTAIN"
    assert bundle.selected_model_id is None
    assert all(
        not item.local_development_admissible for item in bundle.candidates
    )
    assert all(
        item.persistence_relative_improvement == 0.0
        for item in bundle.candidates
    )
    assert next(item for item in bundle.levels if item.level == "L3").status == (
        "FAIL"
    )
    assert next(item for item in bundle.levels if item.level == "L4").status == (
        "FAIL"
    )
    assert bundle.local_l0_l4_complete is False
    assert bundle.scientific_acceptance is False
