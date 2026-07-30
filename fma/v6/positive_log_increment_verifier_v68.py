"""Input-bound recomputation verifier for the V6.8 log-increment pack.

The executor's bundle is treated only as a claim under review.  This module
recompiles the typed IR and recomputes the complete bundle from the frozen
snapshot, thresholds, and replay receipts before returning one level.

Running this code in another process gives context separation, but local
recomputation is still development evidence.  It cannot sign a stage gate,
promote a capability, or grant scientific qualification.
"""

from __future__ import annotations

from .positive_log_increment_v68 import (
    LevelV68,
    PositiveLogIncrementBundleV68,
    PositiveLogIncrementLevelEvidenceV68,
    PositiveLogIncrementModelIRV68,
    PositiveLogIncrementReplayAuthorityV68,
    PositiveLogIncrementReplayReceiptV68,
    PositiveLogIncrementThresholdsV68,
    PositiveScalarSeriesSnapshotV68,
    compile_positive_log_increment_ir_v68,
    execute_positive_log_increment_ir_v68,
)


def recompute_positive_log_increment_level_v68(
    *,
    bundle: PositiveLogIncrementBundleV68,
    level: LevelV68,
    model_ir: PositiveLogIncrementModelIRV68,
    snapshot: PositiveScalarSeriesSnapshotV68,
    thresholds: PositiveLogIncrementThresholdsV68,
    replay_receipts: list[PositiveLogIncrementReplayReceiptV68] | None = None,
    replay_authority: PositiveLogIncrementReplayAuthorityV68 | None = None,
) -> PositiveLogIncrementLevelEvidenceV68:
    """Recompute an exact expected bundle before accepting one level.

    The caller must supply the same frozen inputs that were presented to the
    executor.  A resealed bundle with invented PASS evidence therefore differs
    from the recomputed artifact and is rejected.
    """

    bundle.assert_sealed()
    model_ir.assert_sealed()
    snapshot.assert_sealed()
    thresholds.assert_sealed()
    expected_ir = compile_positive_log_increment_ir_v68(thresholds)
    if model_ir != expected_ir:
        raise ValueError("V6.8 verifier IR differs from the frozen compiler")
    if bundle.model_ir_hash != model_ir.ir_hash:
        raise ValueError("V6.8 verifier bundle is bound to another IR")
    expected = execute_positive_log_increment_ir_v68(
        model_ir=model_ir,
        snapshot=snapshot,
        thresholds=thresholds,
        replay_receipts=replay_receipts,
        replay_authority=replay_authority,
    )
    if bundle != expected:
        raise ValueError(
            "V6.8 log-increment bundle differs from input-bound recomputation"
        )
    evidence = next(item for item in expected.levels if item.level == level)
    evidence.assert_sealed()
    return evidence.model_copy(deep=True)


__all__ = ["recompute_positive_log_increment_level_v68"]
