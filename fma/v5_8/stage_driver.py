"""Role-specific structured-output contracts for the V5.8 S1 runtime."""

from __future__ import annotations

from typing import Any

from fma.hashing import canonical_json
from fma.v5_1.codex_stage_driver import (
    CodexStageRoleTransportV51,
    RoleDraftV51,
    RoleRequestV51,
    _strict_wire_schema,
)


_S1_ARTIFACT_TYPES: dict[str, list[str]] = {
    "s1_prior_model_scout": ["literature_map"],
    "s1_candidate_synthesizer": [
        "selection",
        "selected_candidate_structure",
        "selected_mathematical_form",
        "validation_rule_s3_l0_replay",
        "validation_rule_s3_l1_structural",
        "validation_rule_s3_l2_numerical",
        "validation_rule_s4_l3_holdout",
        "validation_rule_s4_l4_uncertainty",
    ],
    "s1_candidate_synthesizer_repair": [
        "selection",
        "selected_candidate_structure",
        "selected_mathematical_form",
        "validation_rule_s3_l0_replay",
        "validation_rule_s3_l1_structural",
        "validation_rule_s3_l2_numerical",
        "validation_rule_s4_l3_holdout",
        "validation_rule_s4_l4_uncertainty",
    ],
}
_BRANCH_SUFFIXES = ("_blind", "_repair")
_ARRAY_CAPS = {
    "abandon_criteria": 3,
    "applicability_conditions": 2,
    "assumption_ids": 2,
    "assumptions": 2,
    "candidate_family_hints": 8,
    "data_requirement_ids": 4,
    "declared_conservation_laws": 4,
    "declared_limit_cases": 4,
    "identifiability_risks": 4,
    "knowledge_units": 2,
    "limitations": 4,
    "source_unit_ids": 3,
    "symbol_ids": 5,
    "symbols": 5,
    "transfer_assessments": 2,
    "transfer_hypotheses": 2,
    "validation_obligation_ids": 6,
    "validation_rules": 5,
}
_STRING_CAPS = {
    "abandon_criterion": 500,
    "abandon_criteria": 500,
    "falsification_test": 400,
    "failure_consequence": 400,
    "lineage": 800,
    "mathematical_form": 1600,
    "proposed_modification": 400,
    "rationale": 400,
    "required_test": 400,
    "selection_rationale": 1200,
    "statement": 400,
    "target_interpretation": 400,
    "applicability_rule": 1200,
}


def _artifact_types(request: RoleRequestV51) -> list[str] | None:
    if request.stage not in {"S0", "S1"}:
        return None
    if request.role_kind == "reviewer":
        return []
    if request.stage == "S0":
        if request.role_name in {
            "problem_formulator",
            "problem_formulator_repair",
        }:
            return ["decision_function", "regime_diagnosis"]
        return None
    if request.role_name.endswith("_recipient"):
        return ["transfer_assessments"]
    if request.role_name.startswith("s1_cross_paradigm_translator_"):
        return ["transfer_hypotheses"]
    if request.role_name in _S1_ARTIFACT_TYPES:
        return _S1_ARTIFACT_TYPES[request.role_name]
    if request.role_name.endswith(_BRANCH_SUFFIXES):
        return ["assumptions", "candidate", "knowledge_units", "symbols"]
    return None


def _inline_refs(value: object, definitions: dict[str, Any]) -> object:
    if isinstance(value, list):
        return [_inline_refs(item, definitions) for item in value]
    if not isinstance(value, dict):
        return value
    reference = value.get("$ref")
    if isinstance(reference, str) and reference.startswith("#/$defs/"):
        name = reference.removeprefix("#/$defs/")
        return _inline_refs(definitions[name], definitions)
    return {
        key: _inline_refs(item, definitions)
        for key, item in value.items()
        if key != "$defs"
    }


def _bounded_strict_schema(
    schema: dict[str, Any],
    *,
    artifact_type: str,
) -> dict[str, Any]:
    definitions = schema.get("$defs", {})
    inlined = _inline_refs(schema, definitions)
    assert isinstance(inlined, dict)

    def normalize(value: object, name: str) -> None:
        if isinstance(value, list):
            for item in value:
                normalize(item, name)
            return
        if not isinstance(value, dict):
            return
        value.pop("default", None)
        properties = value.get("properties")
        if isinstance(properties, dict):
            value["required"] = sorted(properties)
            value.setdefault("additionalProperties", False)
            for property_name, nested in properties.items():
                normalize(nested, property_name)
        items = value.get("items")
        if isinstance(items, (dict, list)):
            normalize(items, name)
        for union_name in ("anyOf", "oneOf"):
            union = value.get(union_name)
            if isinstance(union, list):
                normalize(union, name)
        if value.get("type") == "string":
            cap = (
                2600
                if artifact_type == "selected_mathematical_form"
                and name == "mathematical_form"
                else _STRING_CAPS.get(name, 240)
            )
            current = value.get("maxLength")
            value["maxLength"] = min(current, cap) if current else cap
        if value.get("type") == "array":
            cap = _ARRAY_CAPS.get(name, _ARRAY_CAPS.get(artifact_type, 4))
            minimum = int(value.get("minItems", 0))
            current = value.get("maxItems")
            if current is not None:
                cap = min(int(current), cap)
            value["maxItems"] = max(minimum, cap)

    normalize(inlined, artifact_type)
    return inlined


def _content_schemas(
    request: RoleRequestV51,
    artifact_types: list[str],
) -> dict[str, dict[str, Any]]:
    public = request.public_inputs
    raw = public.get("required_artifacts")
    if not isinstance(raw, dict):
        raw = public.get("required_artifact")
    if not isinstance(raw, dict):
        raise ValueError("V5.8 role request lacks typed artifact schemas")
    schemas: dict[str, dict[str, Any]] = {}
    for artifact_type in artifact_types:
        artifact_schema = raw.get(artifact_type)
        if not isinstance(artifact_schema, dict):
            raise ValueError(f"V5.8 role request lacks schema for {artifact_type}")
        schemas[artifact_type] = _bounded_strict_schema(
            artifact_schema,
            artifact_type=artifact_type,
        )
    return schemas


def role_draft_schema_v58(request: RoleRequestV51) -> dict[str, Any]:
    """Narrow the generic role envelope to exact typed V5.8 artifacts."""

    schema = _strict_wire_schema(RoleDraftV51)
    properties = schema["properties"]
    properties["rationale"]["maxLength"] = 800
    for field_name, max_items, max_length in (
        ("assumptions", 8, 240),
        ("findings", 8, 400),
        ("uncertainties", 8, 240),
    ):
        properties[field_name]["maxItems"] = max_items
        properties[field_name]["items"]["maxLength"] = max_length
    properties["request_hash"] = {
        "const": request.request_hash,
        "type": "string",
    }
    properties["role_name"] = {
        "const": request.role_name,
        "type": "string",
    }
    properties["authority_claimed"] = {"const": False, "type": "boolean"}
    if request.role_kind == "reviewer":
        properties["verdict"] = {
            "enum": ["APPROVE", "HUMAN", "REJECT"],
            "type": "string",
        }
    else:
        properties["verdict"] = {
            "const": "NOT_APPLICABLE",
            "type": "string",
        }
    if request.allowed_candidate_ids:
        properties["selected_candidate_id"] = {
            "anyOf": [
                {
                    "enum": request.allowed_candidate_ids,
                    "type": "string",
                },
                {"type": "null"},
            ]
        }
    else:
        properties["selected_candidate_id"] = {"type": "null"}

    artifact_types = _artifact_types(request)
    if artifact_types is not None:
        if not artifact_types:
            original = properties["proposed_artifacts"]
            original["maxItems"] = 0
            original["minItems"] = 0
        else:
            content_schemas = _content_schemas(request, artifact_types)
            properties["proposed_artifacts"] = {
                "items": {
                    "anyOf": [
                        {
                            "additionalProperties": False,
                            "properties": {
                                "artifact_type": {
                                    "const": artifact_type,
                                    "type": "string",
                                },
                                "content": content_schemas[artifact_type],
                            },
                            "required": ["artifact_type", "content"],
                            "type": "object",
                        }
                        for artifact_type in artifact_types
                    ]
                },
                "maxItems": len(artifact_types),
                "minItems": len(artifact_types),
                "type": "array",
            }
    return schema


class CodexStageRoleTransportV58(CodexStageRoleTransportV51):
    """V5.1 process isolation with typed, nested V5.8 artifact output."""

    def _output_schema(self, request: RoleRequestV51) -> dict[str, Any]:
        return role_draft_schema_v58(request)

    def _parse_draft(
        self,
        raw: object,
        request: RoleRequestV51,
    ) -> RoleDraftV51:
        artifact_types = _artifact_types(request)
        if artifact_types is None or not isinstance(raw, dict):
            return super()._parse_draft(raw, request)
        normalized = dict(raw)
        proposed = normalized.get("proposed_artifacts")
        if not isinstance(proposed, list):
            return super()._parse_draft(raw, request)
        normalized["proposed_artifacts"] = [
            {
                **item,
                "content": canonical_json(item["content"]),
            }
            if isinstance(item, dict) and not isinstance(item.get("content"), str)
            else item
            for item in proposed
        ]
        return super()._parse_draft(normalized, request)


__all__ = ["CodexStageRoleTransportV58", "role_draft_schema_v58"]
