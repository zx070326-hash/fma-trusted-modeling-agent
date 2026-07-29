"""Input-bound recomputation verifier for the V6.8 scalar ODE wrapper."""

from __future__ import annotations

from fma.v5_2.ode_system import (
    LevelV52,
    ODELevelEvidenceV52,
    ODEThresholdsV52,
    ODETimeSeriesSnapshotV52,
)

from .scalar_ode_pack_v68 import (
    ScalarAutonomousODEBundleV68,
    ScalarAutonomousODEModelIRV68,
    compile_scalar_autonomous_ode_ir_v68,
    execute_scalar_autonomous_ode_ir_v68,
)


def recompute_scalar_autonomous_ode_level_v68(
    *,
    bundle: ScalarAutonomousODEBundleV68,
    level: LevelV52,
    model_ir: ScalarAutonomousODEModelIRV68,
    snapshot: ODETimeSeriesSnapshotV52,
    thresholds: ODEThresholdsV52,
    replay_output_hashes: list[str] | None = None,
) -> ODELevelEvidenceV52:
    """Recompute the complete wrapped bundle before returning one level."""

    bundle.assert_sealed()
    model_ir.assert_sealed()
    snapshot.assert_sealed()
    thresholds.assert_sealed()
    if model_ir != compile_scalar_autonomous_ode_ir_v68(thresholds):
        raise ValueError("V6.8 scalar ODE verifier IR differs")
    expected = execute_scalar_autonomous_ode_ir_v68(
        model_ir=model_ir,
        snapshot=snapshot,
        thresholds=thresholds,
        replay_output_hashes=replay_output_hashes,
    )
    if bundle != expected:
        raise ValueError(
            "V6.8 scalar ODE bundle differs from input-bound recomputation"
        )
    return next(
        item.model_copy(deep=True)
        for item in expected.legacy_bundle.levels
        if item.level == level
    )


__all__ = ["recompute_scalar_autonomous_ode_level_v68"]
