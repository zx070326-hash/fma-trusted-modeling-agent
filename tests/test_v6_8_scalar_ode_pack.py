from __future__ import annotations

import numpy as np
import pytest

from fma.v5_2.ode_system import (
    ODEThresholdsV52,
    ODETimeSeriesSnapshotV52,
    build_ode_bundle_v52,
)
from fma.v6.scalar_ode_pack_v68 import (
    ScalarAutonomousODEBundleV68,
    compile_scalar_autonomous_ode_ir_v68,
    execute_scalar_autonomous_ode_ir_v68,
)
from fma.v6.scalar_ode_verifier_v68 import (
    recompute_scalar_autonomous_ode_level_v68,
)


def _inputs() -> tuple[ODETimeSeriesSnapshotV52, ODEThresholdsV52]:
    times = np.arange(30, dtype=float)
    values = 20.0 * np.exp(0.025 * times)
    return (
        ODETimeSeriesSnapshotV52.seal(
            task_id="v68-scalar-ode-wrapper",
            time_unit="day",
            state_unit="registered_positive_state",
            times=times.tolist(),
            observations=values.tolist(),
            source_id="v68-wrapper-fixture",
            fixture_only=True,
        ),
        ODEThresholdsV52.seal(),
    )


def test_scalar_ode_v68_ir_executes_the_unmodified_v52_kernel() -> None:
    snapshot, thresholds = _inputs()
    model_ir = compile_scalar_autonomous_ode_ir_v68(thresholds)
    wrapped = execute_scalar_autonomous_ode_ir_v68(
        model_ir=model_ir,
        snapshot=snapshot,
        thresholds=thresholds,
    )
    direct = build_ode_bundle_v52(
        snapshot=snapshot,
        thresholds=thresholds,
    )

    model_ir.assert_sealed()
    wrapped.assert_sealed()
    assert wrapped.legacy_bundle == direct
    assert wrapped.model_ir_hash == model_ir.ir_hash
    assert wrapped.scientific_acceptance is False
    assert wrapped.scientific_qualification_granted is False
    assert wrapped.real_world_capability_established is False
    assert wrapped.real_world_action_authorized is False


def test_scalar_ode_v68_verifier_recomputes_from_frozen_inputs() -> None:
    snapshot, thresholds = _inputs()
    model_ir = compile_scalar_autonomous_ode_ir_v68(thresholds)
    wrapped = execute_scalar_autonomous_ode_ir_v68(
        model_ir=model_ir,
        snapshot=snapshot,
        thresholds=thresholds,
    )
    level = recompute_scalar_autonomous_ode_level_v68(
        bundle=wrapped,
        level="L2",
        model_ir=model_ir,
        snapshot=snapshot,
        thresholds=thresholds,
    )
    assert level == next(
        item for item in wrapped.legacy_bundle.levels if item.level == "L2"
    )

    payload = wrapped.model_dump(
        mode="json",
        exclude={"model_ir_hash", "bundle_hash"},
    )
    resealed = ScalarAutonomousODEBundleV68.seal(
        **payload,
        model_ir_hash="f" * 64,
    )
    resealed.assert_sealed()
    with pytest.raises(ValueError, match="input-bound recomputation"):
        recompute_scalar_autonomous_ode_level_v68(
            bundle=resealed,
            level="L2",
            model_ir=model_ir,
            snapshot=snapshot,
            thresholds=thresholds,
        )


def test_scalar_ode_v68_rejects_tampered_ir() -> None:
    snapshot, thresholds = _inputs()
    model_ir = compile_scalar_autonomous_ode_ir_v68(thresholds)
    tampered = model_ir.model_copy(update={"threshold_hash": "e" * 64})
    with pytest.raises(ValueError, match="IR"):
        execute_scalar_autonomous_ode_ir_v68(
            model_ir=tampered,
            snapshot=snapshot,
            thresholds=thresholds,
        )
