"""Additive V5.1 execution and scientific-evaluation layer.

V5.1 consumes V5.0 artifacts without changing their interpretation.
"""

from .codex_stage_driver import (
    CodexStageRoleTransportV51,
    FixtureStageRoleTransportV51,
    RoleDraftV51,
    RoleProcessOutcomeV51,
    RoleRequestV51,
    StageRoleDriverV51,
)
from .evaluation_harness import (
    AblationComparisonV51,
    GoldStagePackageV51,
    MechanismProfileV51,
    compare_ablation_runs_v51,
)

__all__ = [
    "AblationComparisonV51",
    "CodexStageRoleTransportV51",
    "FixtureStageRoleTransportV51",
    "GoldStagePackageV51",
    "MechanismProfileV51",
    "RoleDraftV51",
    "RoleProcessOutcomeV51",
    "RoleRequestV51",
    "StageRoleDriverV51",
    "compare_ablation_runs_v51",
]
