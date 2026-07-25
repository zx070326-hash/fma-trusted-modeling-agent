from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Literal

import numpy as np
from pydantic import Field, model_validator

from fma.hashing import sha256_value
from fma.schemas import StrictModel

from .empirical_schemas import TimeSeriesSnapshot
from .schemas import Identifier, Sha256, _assert_timezone


NonNegativeFinite = Annotated[float, Field(ge=0, allow_inf_nan=False)]


def _hash_without(model: StrictModel, field: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude={field}))


class WindowShiftSpec(StrictModel):
    schema_version: Literal["2.1"] = "2.1"
    diagnostic_id: Identifier
    data_snapshot_hash: Sha256
    holdout_points: Annotated[int, Field(ge=4)]
    reference_points: Annotated[int, Field(ge=4)]
    max_standardized_mean_shift: Annotated[
        float, Field(gt=0, allow_inf_nan=False)
    ]
    minimum_scale_ratio: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    maximum_scale_ratio: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    max_reference_range_exceedance: Annotated[
        float, Field(ge=0, le=1, allow_inf_nan=False)
    ]
    frozen_at: datetime
    shift_spec_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_spec(self) -> "WindowShiftSpec":
        _assert_timezone(self.frozen_at, "frozen_at")
        if self.minimum_scale_ratio >= self.maximum_scale_ratio:
            raise ValueError("scale ratio bounds are reversed")
        if self.shift_spec_hash and self.shift_spec_hash != self.content_hash():
            raise ValueError("shift_spec_hash does not match shift spec")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "shift_spec_hash")

    def assert_sealed(self) -> None:
        if not self.shift_spec_hash or self.shift_spec_hash != self.content_hash():
            raise ValueError("window shift spec is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "WindowShiftSpec":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"shift_spec_hash"}),
            shift_spec_hash=draft.content_hash(),
        )


class WindowShiftReport(StrictModel):
    schema_version: Literal["2.1"] = "2.1"
    report_id: Identifier
    data_snapshot_hash: Sha256
    shift_spec_hash: Sha256
    reference_mean: NonNegativeFinite
    holdout_mean: NonNegativeFinite
    standardized_mean_shift: NonNegativeFinite | None
    scale_ratio: NonNegativeFinite | None
    reference_range_exceedance: Annotated[
        float, Field(ge=0, le=1, allow_inf_nan=False)
    ]
    status: Literal["stable_by_frozen_diagnostics", "shift_detected"]
    reason_codes: list[Literal[
        "mean_shift_exceeded",
        "scale_shift_exceeded",
        "reference_range_exceedance",
        "reference_scale_degenerate",
    ]]
    evaluated_at: datetime
    report_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_report(self) -> "WindowShiftReport":
        _assert_timezone(self.evaluated_at, "evaluated_at")
        if (self.status == "stable_by_frozen_diagnostics") == bool(self.reason_codes):
            raise ValueError("stable reports need no reasons; detected shifts need reasons")
        if self.report_hash and self.report_hash != self.content_hash():
            raise ValueError("report_hash does not match shift report")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "report_hash")

    def assert_sealed(self) -> None:
        if not self.report_hash or self.report_hash != self.content_hash():
            raise ValueError("window shift report is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "WindowShiftReport":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"report_hash"}),
            report_hash=draft.content_hash(),
        )


class WindowShiftEvaluator:
    def evaluate(
        self,
        snapshot: TimeSeriesSnapshot,
        spec: WindowShiftSpec,
        *,
        evaluated_at: datetime | None = None,
    ) -> WindowShiftReport:
        snapshot.assert_sealed()
        spec.assert_sealed()
        if spec.data_snapshot_hash != snapshot.snapshot_hash:
            raise ValueError("shift spec is bound to another data snapshot")
        if spec.holdout_points + spec.reference_points > len(snapshot.points):
            raise ValueError("shift windows exceed the available observations")
        values = np.asarray([point.value for point in snapshot.points], dtype=float)
        holdout = values[-spec.holdout_points :]
        reference = values[
            -(spec.holdout_points + spec.reference_points) : -spec.holdout_points
        ]
        reference_mean = float(reference.mean())
        holdout_mean = float(holdout.mean())
        reference_scale = float(reference.std(ddof=1))
        holdout_scale = float(holdout.std(ddof=1))
        reasons: list[str] = []
        if reference_scale <= 1e-15:
            standardized_mean_shift = None
            scale_ratio = None
            if abs(holdout_mean - reference_mean) > 1e-12 or holdout_scale > 1e-12:
                reasons.append("reference_scale_degenerate")
        else:
            standardized_mean_shift = abs(holdout_mean - reference_mean) / reference_scale
            scale_ratio = holdout_scale / reference_scale
            if standardized_mean_shift > spec.max_standardized_mean_shift:
                reasons.append("mean_shift_exceeded")
            if not spec.minimum_scale_ratio <= scale_ratio <= spec.maximum_scale_ratio:
                reasons.append("scale_shift_exceeded")
        outside_fraction = float(
            np.mean((holdout < reference.min()) | (holdout > reference.max()))
        )
        if outside_fraction > spec.max_reference_range_exceedance:
            reasons.append("reference_range_exceedance")
        assert snapshot.snapshot_hash is not None
        assert spec.shift_spec_hash is not None
        return WindowShiftReport.seal(
            report_id="adjacent_window_shift_diagnostic",
            data_snapshot_hash=snapshot.snapshot_hash,
            shift_spec_hash=spec.shift_spec_hash,
            reference_mean=reference_mean,
            holdout_mean=holdout_mean,
            standardized_mean_shift=standardized_mean_shift,
            scale_ratio=scale_ratio,
            reference_range_exceedance=outside_fraction,
            status="shift_detected" if reasons else "stable_by_frozen_diagnostics",
            reason_codes=list(dict.fromkeys(reasons)),
            evaluated_at=evaluated_at or datetime.now(timezone.utc),
        )
