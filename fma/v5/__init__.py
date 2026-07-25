"""FMA V5: graph-native S0--S6 workspaces and external evaluation.

V5 is additive.  It does not change V1--V4 artifacts, hashes, promotion
semantics, or scientific authority.
"""

from .scaffold import (
    WorkspaceScaffoldError,
    scaffold_task_workspace,
    validate_task_scaffold,
)
from .check_registry import (
    AdapterContextV50,
    AdapterOutcomeV50,
    CheckRegistryV50,
    ScientificCheckAdapterV50,
)
from .stage_workspace import (
    POLICIES,
    STAGES,
    StageWorkspaceError,
    StageWorkspaceV50,
)
from .workspace_schemas import (
    AdapterExecutionReceiptV50,
    CheckResultV50,
    GateCertificateV50,
    GateEvaluationV50,
    IndependentReviewReceiptV50,
    PredictionSealV50,
    RoleExecutionReceiptV50,
    TaskWorkspaceSpecV50,
    WorkflowProfileV50,
    WorkflowStatusV50,
)

__all__ = [
    "AdapterExecutionReceiptV50",
    "CheckResultV50",
    "CheckRegistryV50",
    "GateCertificateV50",
    "GateEvaluationV50",
    "IndependentReviewReceiptV50",
    "PredictionSealV50",
    "RoleExecutionReceiptV50",
    "POLICIES",
    "STAGES",
    "StageWorkspaceError",
    "StageWorkspaceV50",
    "TaskWorkspaceSpecV50",
    "WorkflowProfileV50",
    "WorkflowStatusV50",
    "WorkspaceScaffoldError",
    "AdapterContextV50",
    "AdapterOutcomeV50",
    "ScientificCheckAdapterV50",
    "scaffold_task_workspace",
    "validate_task_scaffold",
]
