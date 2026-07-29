"""Typed V6.8 execution wrapper around the historical V5.2 scalar ODE pack.

The wrapper is additive: it does not reinterpret or modify the V5.2 bundle.
It binds a fixed, observation-free V6.8 IR to the exact legacy inputs and
output, and keeps all V6.8 qualification/action flags false.
"""

from __future__ import annotations

from typing import Literal

from pydantic import model_validator

from fma.hashing import sha256_value
from fma.schemas import StrictModel
from fma.v2.schemas import Sha256
from fma.v5_2.ode_system import (
    ODEScientificBundleV52,
    ODEThresholdsV52,
    ODETimeSeriesSnapshotV52,
    build_ode_bundle_v52,
)


_FAMILIES = ["constant", "exponential", "gompertz", "logistic"]


def _hash_without(model: StrictModel, *fields: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude=set(fields)))


class ScalarAutonomousODEModelIRV68(StrictModel):
    """Observation-free intent for the exact V5.2 family registry."""

    schema_version: Literal["6.8-scalar-autonomous-ode-ir"] = (
        "6.8-scalar-autonomous-ode-ir"
    )
    capability_pack_id: Literal["scalar_autonomous_ode_v52"] = (
        "scalar_autonomous_ode_v52"
    )
    candidate_families: list[
        Literal["constant", "exponential", "gompertz", "logistic"]
    ]
    baseline_ids: list[Literal["constant", "persistence"]]
    forecast_horizon_steps: Literal[1] = 1
    threshold_hash: Sha256
    model_text_executable: Literal[False] = False
    arbitrary_code_execution_permitted: Literal[False] = False
    causal_interpretation_permitted: Literal[False] = False
    ir_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_ir(self) -> "ScalarAutonomousODEModelIRV68":
        if self.candidate_families != _FAMILIES:
            raise ValueError("V6.8 scalar ODE IR family registry differs")
        if self.baseline_ids != ["constant", "persistence"]:
            raise ValueError("V6.8 scalar ODE IR baseline registry differs")
        if self.ir_hash and self.ir_hash != self.content_hash():
            raise ValueError("V6.8 scalar ODE IR hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "ir_hash")

    def assert_sealed(self) -> None:
        type(self).model_validate(self.model_dump(mode="json"))
        if not self.ir_hash or self.ir_hash != self.content_hash():
            raise ValueError("V6.8 scalar ODE IR is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ScalarAutonomousODEModelIRV68":
        draft = cls(**data)
        payload = draft.model_dump(mode="json", exclude={"ir_hash"})
        payload["ir_hash"] = draft.content_hash()
        return cls(**payload)


class ScalarAutonomousODEBundleV68(StrictModel):
    """Exact IR binding around one untouched V5.2 result."""

    schema_version: Literal["6.8-scalar-autonomous-ode-bundle"] = (
        "6.8-scalar-autonomous-ode-bundle"
    )
    model_ir_hash: Sha256
    snapshot_hash: Sha256
    threshold_hash: Sha256
    legacy_bundle: ODEScientificBundleV52
    local_legacy_l0_l4_complete: bool
    scientific_acceptance: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_capability_established: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    bundle_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_bundle(self) -> "ScalarAutonomousODEBundleV68":
        if (
            self.snapshot_hash != self.legacy_bundle.snapshot_hash
            or self.threshold_hash != self.legacy_bundle.threshold_hash
        ):
            raise ValueError("V6.8 scalar ODE wrapper input binding differs")
        if (
            self.local_legacy_l0_l4_complete
            != self.legacy_bundle.scientific_acceptance
        ):
            raise ValueError("V6.8 scalar ODE local completion differs")
        if self.bundle_hash and self.bundle_hash != self.content_hash():
            raise ValueError("V6.8 scalar ODE bundle hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "bundle_hash")

    def assert_sealed(self) -> None:
        type(self).model_validate(self.model_dump(mode="json"))
        if not self.bundle_hash or self.bundle_hash != self.content_hash():
            raise ValueError("V6.8 scalar ODE bundle is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ScalarAutonomousODEBundleV68":
        draft = cls(**data)
        payload = draft.model_dump(mode="json", exclude={"bundle_hash"})
        payload["bundle_hash"] = draft.content_hash()
        return cls(**payload)


def compile_scalar_autonomous_ode_ir_v68(
    thresholds: ODEThresholdsV52,
) -> ScalarAutonomousODEModelIRV68:
    """Compile the exact legacy registry without accepting observations."""

    thresholds.assert_sealed()
    return ScalarAutonomousODEModelIRV68.seal(
        candidate_families=list(_FAMILIES),
        baseline_ids=["constant", "persistence"],
        threshold_hash=str(thresholds.threshold_hash),
    )


def execute_scalar_autonomous_ode_ir_v68(
    *,
    model_ir: ScalarAutonomousODEModelIRV68,
    snapshot: ODETimeSeriesSnapshotV52,
    thresholds: ODEThresholdsV52,
    replay_output_hashes: list[str] | None = None,
) -> ScalarAutonomousODEBundleV68:
    """Execute one exact V6.8 IR through the unmodified V5.2 kernel."""

    model_ir.assert_sealed()
    snapshot.assert_sealed()
    thresholds.assert_sealed()
    if model_ir != compile_scalar_autonomous_ode_ir_v68(thresholds):
        raise ValueError("V6.8 scalar ODE IR differs from the compiler")
    legacy = build_ode_bundle_v52(
        snapshot=snapshot,
        thresholds=thresholds,
        replay_output_hashes=replay_output_hashes,
    )
    return ScalarAutonomousODEBundleV68.seal(
        model_ir_hash=str(model_ir.ir_hash),
        snapshot_hash=str(snapshot.snapshot_hash),
        threshold_hash=str(thresholds.threshold_hash),
        legacy_bundle=legacy,
        local_legacy_l0_l4_complete=legacy.scientific_acceptance,
    )


__all__ = [
    "ScalarAutonomousODEBundleV68",
    "ScalarAutonomousODEModelIRV68",
    "compile_scalar_autonomous_ode_ir_v68",
    "execute_scalar_autonomous_ode_ir_v68",
]
