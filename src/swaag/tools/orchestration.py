from __future__ import annotations

import json
from typing import Any, cast

from swaag.orchestration_api import OrchestrationApi
from swaag.tools.base import Tool, ToolContext, ToolValidationError
from swaag.types import ToolExecutionResult, ToolKind
from swaag.utils import stable_json_dumps


def _nullable(schema: dict[str, Any]) -> dict[str, Any]:
    return {"anyOf": [schema, {"type": "null"}]}


_PLAN_NODE_SCHEMA = {
    "type": "object",
    "properties": {
        "key": {"type": "string"},
        "completion_mode": {"type": "string", "enum": ["natural", "continuous"]},
        "objective": {"type": "string"},
        "priority": {"type": "number"},
        "model_key": _nullable({"type": "string"}),
        "finish_criteria": _nullable({"type": "string"}),
        "abort_criteria": _nullable({"type": "string"}),
        "resources_json": _nullable({"type": "string"}),
    },
    "required": [
        "key",
        "completion_mode",
        "objective",
        "priority",
        "model_key",
        "finish_criteria",
        "abort_criteria",
        "resources_json",
    ],
    "additionalProperties": False,
}

_PLAN_EDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "source": {"type": "string"},
        "target": {"type": "string"},
        "when": {
            "type": "string",
            "enum": ["completed", "terminal", "failed", "semantic"],
        },
        "semantic_question": _nullable({"type": "string"}),
        "input_mapping_json": _nullable({"type": "string"}),
    },
    "required": [
        "source",
        "target",
        "when",
        "semantic_question",
        "input_mapping_json",
    ],
    "additionalProperties": False,
}

_PLAN_SPEC_SCHEMA = {
    "type": "object",
    "properties": {
        "objective": {"type": "string"},
        "scheduling_mode": {
            "type": "string",
            "enum": ["parallel", "sequential"],
        },
        "max_parallel": {"type": "integer"},
        "reporting_mode": {
            "type": "string",
            "enum": ["all", "terminal", "important", "completion_only", "manual"],
        },
        "resource_limits_json": _nullable({"type": "string"}),
        "nodes": {"type": "array", "items": _PLAN_NODE_SCHEMA},
        "edges": {"type": "array", "items": _PLAN_EDGE_SCHEMA},
        "start": {"type": "boolean"},
    },
    "required": [
        "objective",
        "scheduling_mode",
        "max_parallel",
        "reporting_mode",
        "resource_limits_json",
        "nodes",
        "edges",
        "start",
    ],
    "additionalProperties": False,
}


def _decode_object_json(value: str | None, *, field: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, str):
        raise ToolValidationError(f"orchestration_control.{field} must be JSON string or null")
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ToolValidationError(
            f"orchestration_control.{field} is invalid JSON: {exc}"
        ) from exc
    if not isinstance(decoded, dict):
        raise ToolValidationError(
            f"orchestration_control.{field} must encode an object"
        )
    return decoded


def _validate_plan_spec(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ToolValidationError("orchestration_control.plan_spec must be object or null")
    expected = set(_PLAN_SPEC_SCHEMA["required"])
    if set(raw) != expected:
        raise ToolValidationError(
            "orchestration_control.plan_spec requires exactly: "
            + ", ".join(sorted(expected))
        )
    objective = raw["objective"]
    if not isinstance(objective, str) or not objective.strip():
        raise ToolValidationError("orchestration_control.plan_spec.objective must be non-empty")
    scheduling_mode = raw["scheduling_mode"]
    if scheduling_mode not in {"parallel", "sequential"}:
        raise ToolValidationError("orchestration_control.plan_spec.scheduling_mode is invalid")
    maximum = raw["max_parallel"]
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 0:
        raise ToolValidationError(
            "orchestration_control.plan_spec.max_parallel must be non-negative"
        )
    reporting_mode = raw["reporting_mode"]
    if reporting_mode not in {"all", "terminal", "important", "completion_only", "manual"}:
        raise ToolValidationError("orchestration_control.plan_spec.reporting_mode is invalid")
    if not isinstance(raw["start"], bool):
        raise ToolValidationError("orchestration_control.plan_spec.start must be boolean")
    nodes = raw["nodes"]
    if not isinstance(nodes, list) or not nodes:
        raise ToolValidationError(
            "orchestration_control.plan_spec.nodes must be a non-empty array"
        )
    normalized_nodes: list[dict[str, Any]] = []
    node_expected = set(_PLAN_NODE_SCHEMA["required"])
    for index, node in enumerate(nodes):
        if isinstance(node, dict):
            node = {"completion_mode": "natural", **node}
        if not isinstance(node, dict) or set(node) != node_expected:
            raise ToolValidationError(
                f"orchestration_control.plan_spec.nodes[{index}] must use the exact node schema"
            )
        if node["completion_mode"] not in {"natural", "continuous"}:
            raise ToolValidationError("Invalid completion_mode")
        key = node["key"]
        node_objective = node["objective"]
        priority = node["priority"]
        if not isinstance(key, str) or not key.strip():
            raise ToolValidationError(f"orchestration_control.plan_spec.nodes[{index}].key must be non-empty")
        if not isinstance(node_objective, str) or not node_objective.strip():
            raise ToolValidationError(f"orchestration_control.plan_spec.nodes[{index}].objective must be non-empty")
        if isinstance(priority, bool) or not isinstance(priority, (int, float)) or float(priority) <= 0:
            raise ToolValidationError(f"orchestration_control.plan_spec.nodes[{index}].priority must be positive")
        for field in ("model_key", "finish_criteria", "abort_criteria"):
            value = node[field]
            if value is not None and not isinstance(value, str):
                raise ToolValidationError(
                    f"orchestration_control.plan_spec.nodes[{index}].{field} must be string or null"
                )
        normalized_nodes.append(
            {
                "key": key.strip(),
                "completion_mode": node["completion_mode"],
                "objective": node_objective,
                "priority": float(priority),
                "model_key": node["model_key"].strip() if isinstance(node["model_key"], str) and node["model_key"].strip() else None,
                "finish_criteria": node["finish_criteria"] if isinstance(node["finish_criteria"], str) and node["finish_criteria"].strip() else None,
                "abort_criteria": node["abort_criteria"] if isinstance(node["abort_criteria"], str) and node["abort_criteria"].strip() else None,
                "resources": _decode_object_json(
                    node["resources_json"],
                    field=f"plan_spec.nodes[{index}].resources_json",
                ),
            }
        )
    edges = raw["edges"]
    if not isinstance(edges, list):
        raise ToolValidationError("orchestration_control.plan_spec.edges must be an array")
    edge_expected = set(_PLAN_EDGE_SCHEMA["required"])
    normalized_edges: list[dict[str, Any]] = []
    for index, edge in enumerate(edges):
        if not isinstance(edge, dict) or set(edge) != edge_expected:
            raise ToolValidationError(
                f"orchestration_control.plan_spec.edges[{index}] must use the exact edge schema"
            )
        source = edge["source"]
        target = edge["target"]
        when = edge["when"]
        question = edge["semantic_question"]
        if not isinstance(source, str) or not source.strip() or not isinstance(target, str) or not target.strip():
            raise ToolValidationError(
                f"orchestration_control.plan_spec.edges[{index}] source/target must be non-empty"
            )
        if when not in {"completed", "terminal", "failed", "semantic"}:
            raise ToolValidationError(
                f"orchestration_control.plan_spec.edges[{index}].when is invalid"
            )
        if question is not None and not isinstance(question, str):
            raise ToolValidationError(
                f"orchestration_control.plan_spec.edges[{index}].semantic_question must be string or null"
            )
        condition: dict[str, Any] = {"when": when}
        if when == "semantic" and isinstance(question, str) and question.strip():
            condition["question"] = question
        normalized_edges.append(
            {
                "source": source.strip(),
                "target": target.strip(),
                "condition": condition,
                "input_mapping": _decode_object_json(
                    edge["input_mapping_json"],
                    field=f"plan_spec.edges[{index}].input_mapping_json",
                ),
            }
        )
    return {
        "objective": objective,
        "scheduling_mode": scheduling_mode,
        "max_parallel": maximum,
        "reporting_mode": reporting_mode,
        "resource_limits": _decode_object_json(
            raw["resource_limits_json"], field="plan_spec.resource_limits_json"
        ),
        "nodes": normalized_nodes,
        "edges": normalized_edges,
        "start": raw["start"],
    }


class OrchestrationControlTool(Tool):
    name = "orchestration_control"
    description = (
        "Create, inspect, revise, execute, and cancel the durable background-worker "
        "dependency graph owned by the user-facing SWAAG orchestrator."
    )
    usage_guidance = (
        "Use this for orchestration, not ordinary worker task work. Semantically decide "
        "the plan first. When the complete graph is already known, prefer plan.apply with symbolic node keys so the full draft can be validated and optionally started in one call; otherwise encode nodes and dependencies incrementally. Read the "
        "current plan before revising it. Mechanical dependency predicates may be "
        "executed directly; semantic branch conditions remain your decision."
    )
    kind = "stateful"
    required_runtime_capability = "orchestration"
    input_schema = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": [
                    "supervision",
                    "plan.apply",
                    "create",
                    "configure",
                    "reporting.configure",
                    "list",
                    "get",
                    "validate",
                    "node.add",
                    "node.revise",
                    "dependency.add",
                    "dependency.resolve",
                    "dependency.remove",
                    "start",
                    "advance",
                    "cancel",
                    "notify",
                    "notifications",
                    "notifications.wait",
                    "notification.ack",
                ],
            },
            "completion_mode": _nullable({"type": "string", "enum": ["natural", "continuous"]}),
            "plan_id": _nullable({"type": "string"}),
            "plan_spec": _nullable(_PLAN_SPEC_SCHEMA),
            "objective": _nullable({"type": "string"}),
            "scheduling_mode": _nullable({"type": "string", "enum": ["parallel", "sequential"]}),
            "reporting_mode": _nullable({"type": "string", "enum": ["all", "terminal", "important", "completion_only", "manual"]}),
            "max_parallel": _nullable({"type": "integer"}),
            "resource_limits_json": _nullable({"type": "string"}),
            "resources_json": _nullable({"type": "string"}),
            "node_id": _nullable({"type": "string"}),
            "source_node_id": _nullable({"type": "string"}),
            "target_node_id": _nullable({"type": "string"}),
            "edge_id": _nullable({"type": "string"}),
            "satisfied": _nullable({"type": "boolean"}),
            "decision": _nullable({"type": "string"}),
            "priority": _nullable({"type": "number"}),
            "model_key": _nullable({"type": "string"}),
            "finish_criteria": _nullable({"type": "string"}),
            "abort_criteria": _nullable({"type": "string"}),
            "replace_worker": _nullable({"type": "boolean"}),
            "condition_json": _nullable({"type": "string"}),
            "input_mapping_json": _nullable({"type": "string"}),
            "reason": _nullable({"type": "string"}),
            "notification_kind": _nullable({"type": "string"}),
            "notification_importance": _nullable({"type": "string", "enum": ["routine", "normal", "important", "critical"]}),
            "notification_id": _nullable({"type": "string"}),
            "notification_payload_json": _nullable({"type": "string"}),
            "after_sequence": _nullable({"type": "integer"}),
            "unacknowledged_only": _nullable({"type": "boolean"}),
            "timeout_seconds": _nullable({"type": "number"}),
        },
        "required": [
            "operation",
            "completion_mode",
            "plan_id",
            "plan_spec",
            "objective",
            "scheduling_mode",
            "reporting_mode",
            "max_parallel",
            "resource_limits_json",
            "resources_json",
            "node_id",
            "source_node_id",
            "target_node_id",
            "edge_id",
            "satisfied",
            "decision",
            "priority",
            "model_key",
            "finish_criteria",
            "abort_criteria",
            "replace_worker",
            "condition_json",
            "input_mapping_json",
            "reason",
            "notification_kind",
            "notification_importance",
            "notification_id",
            "notification_payload_json",
            "after_sequence",
            "unacknowledged_only",
            "timeout_seconds",        ],
        "additionalProperties": False,
    }

    def effective_kind(self, validated_input: dict[str, Any]) -> ToolKind:
        operation = validated_input["operation"]
        if operation in {
            "supervision",
            "list",
            "get",
            "validate",
            "notifications",
            "notifications.wait",
        }:
            return "pure"
        if operation in {"start", "advance", "cancel"}:
            return "side_effect"
        if operation == "plan.apply":
            spec = validated_input.get("plan_spec") or {}
            if isinstance(spec, dict) and spec.get("start") is True:
                return "side_effect"
        return "stateful"

    def validate(self, raw_input: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(raw_input, dict):
            raise ToolValidationError("orchestration_control input must be an object")
        raw_input = {"completion_mode": None, **raw_input}
        if raw_input["completion_mode"] not in (None, "natural", "continuous"):
            raise ToolValidationError("Invalid completion_mode")
        expected = set(self.input_schema["required"])
        if set(raw_input) != expected:
            raise ToolValidationError(
                "orchestration_control requires exactly: " + ", ".join(sorted(expected))
            )
        operation = raw_input.get("operation")
        allowed = set(self.input_schema["properties"]["operation"]["enum"])
        if operation not in allowed:
            raise ToolValidationError("orchestration_control.operation is invalid")
        result = dict(raw_input)
        semantic_text_fields = {
            "objective",
            "decision",
            "finish_criteria",
            "abort_criteria",
            "reason",
        }
        for key in (
            "plan_id",
            "objective",
            "scheduling_mode",
            "reporting_mode",
            "node_id",
            "source_node_id",
            "target_node_id",
            "edge_id",
            "decision",
            "model_key",
            "finish_criteria",
            "abort_criteria",
            "reason",
            "notification_kind",
            "notification_importance",
            "notification_id",
        ):
            value = result[key]
            if value is not None and not isinstance(value, str):
                raise ToolValidationError(f"orchestration_control.{key} must be string or null")
            if isinstance(value, str):
                result[key] = (
                    value if key in semantic_text_fields and value.strip()
                    else (value.strip() or None)
                )
        after_sequence = result["after_sequence"]
        if after_sequence is not None and (
            isinstance(after_sequence, bool)
            or not isinstance(after_sequence, int)
            or after_sequence < 0
        ):
            raise ToolValidationError(
                "orchestration_control.after_sequence must be non-negative or null"
            )
        unacknowledged = result["unacknowledged_only"]
        if unacknowledged is not None and not isinstance(unacknowledged, bool):
            raise ToolValidationError(
                "orchestration_control.unacknowledged_only must be boolean or null"
            )
        timeout = result["timeout_seconds"]
        if timeout is not None and (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not 0 <= float(timeout) <= 60
        ):
            raise ToolValidationError(
                "orchestration_control.timeout_seconds must be between 0 and 60 or null"
            )
        if timeout is not None:
            result["timeout_seconds"] = float(timeout)
        maximum = result["max_parallel"]
        if maximum is not None and (
            isinstance(maximum, bool)
            or not isinstance(maximum, int)
            or maximum < 0
        ):
            raise ToolValidationError(
                "orchestration_control.max_parallel must be non-negative or null"
            )
        priority = result["priority"]
        if priority is not None and (
            isinstance(priority, bool)
            or not isinstance(priority, (int, float))
            or float(priority) <= 0
        ):
            raise ToolValidationError("orchestration_control.priority must be positive or null")
        if priority is not None:
            result["priority"] = float(priority)
        satisfied = result["satisfied"]
        if satisfied is not None and not isinstance(satisfied, bool):
            raise ToolValidationError(
                "orchestration_control.satisfied must be boolean or null"
            )
        replace_worker = result["replace_worker"]
        if replace_worker is not None and not isinstance(replace_worker, bool):
            raise ToolValidationError(
                "orchestration_control.replace_worker must be boolean or null"
            )
        result["plan_spec"] = _validate_plan_spec(result["plan_spec"])
        for key in (
            "condition_json",
            "input_mapping_json",
            "notification_payload_json",
            "resource_limits_json",
            "resources_json",
        ):
            value = result[key]
            if value is None:
                result[key.removesuffix("_json")] = None
                continue
            if not isinstance(value, str):
                raise ToolValidationError(f"orchestration_control.{key} must be JSON string or null")
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ToolValidationError(f"orchestration_control.{key} is invalid JSON: {exc}") from exc
            if not isinstance(decoded, dict):
                raise ToolValidationError(f"orchestration_control.{key} must encode an object")
            result[key.removesuffix("_json")] = decoded
        return result

    def execute(self, validated_input: dict[str, Any], context: ToolContext) -> ToolExecutionResult:
        capability = context.runtime_capabilities.get("orchestration")
        if capability is None:
            raise RuntimeError("No orchestration control channel is bound to this session")
        api = cast(OrchestrationApi, capability)
        operation = validated_input["operation"]
        payload: dict[str, Any] = {}
        if validated_input.get("completion_mode") is not None:
            if operation != "node.add":
                raise ToolValidationError("completion_mode is only for node.add; set it in plan_spec nodes for plan.apply")
            payload["completion_mode"] = validated_input["completion_mode"]
        for key in (
            "plan_id",
            "objective",
            "scheduling_mode",
            "reporting_mode",
            "max_parallel",
            "node_id",
            "source_node_id",
            "target_node_id",
            "edge_id",
            "satisfied",
            "decision",
            "priority",
            "model_key",
            "finish_criteria",
            "abort_criteria",
            "replace_worker",
            "reason",
            "notification_id",
            "after_sequence",
            "unacknowledged_only",
            "timeout_seconds",
        ):
            if validated_input.get(key) is not None:
                payload[key] = validated_input[key]
        if validated_input.get("plan_spec") is not None:
            payload["plan_spec"] = validated_input["plan_spec"]
        if validated_input.get("resource_limits") is not None:
            payload["resource_limits"] = validated_input["resource_limits"]
        if validated_input.get("resources") is not None:
            payload["resources"] = validated_input["resources"]
        if validated_input.get("condition") is not None:
            payload["condition"] = validated_input["condition"]
        if validated_input.get("input_mapping") is not None:
            payload["input_mapping"] = validated_input["input_mapping"]
        if operation == "notify":
            if validated_input.get("notification_kind") is not None:
                payload["kind"] = validated_input["notification_kind"]
            if validated_input.get("notification_importance") is not None:
                payload["importance"] = validated_input["notification_importance"]
            if validated_input.get("notification_payload") is not None:
                payload["payload"] = validated_input["notification_payload"]
        output = api.execute(operation, payload)
        return ToolExecutionResult(
            self.name,
            output,
            "orchestration_control result: " + stable_json_dumps(output, indent=2),
        )


ORCHESTRATION_TOOLS = [OrchestrationControlTool()]
