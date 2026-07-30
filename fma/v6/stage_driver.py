"""V6.6 role transport constraints layered over the V5.8 wire protocol."""

from __future__ import annotations

from typing import Any

from fma.v5_1.codex_stage_driver import RoleRequestV51
from fma.v5_8.stage_driver import (
    CodexStageRoleTransportV58,
    role_draft_schema_v58,
)
from fma.v6.stage_review_recovery import S0ReviewDefectCodeV66


def role_draft_schema_v66(request: RoleRequestV51) -> dict[str, Any]:
    """Constrain S0 referee findings to code-owned, non-prose defect codes."""

    schema = role_draft_schema_v58(request)
    if request.stage == "S0" and request.role_kind == "reviewer":
        findings = schema["properties"]["findings"]
        findings["maxItems"] = 16
        findings["items"] = {
            "enum": [item.value for item in S0ReviewDefectCodeV66],
            "type": "string",
        }
    return schema


class CodexStageRoleTransportV66(CodexStageRoleTransportV58):
    """Keep V5.8 process isolation while narrowing S0 reviewer feedback."""

    def _output_schema(self, request: RoleRequestV51) -> dict[str, Any]:
        return role_draft_schema_v66(request)


__all__ = ["CodexStageRoleTransportV66", "role_draft_schema_v66"]
