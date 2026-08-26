from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping

from swaag.schema_portability import assert_portable_json_schema
from swaag.tools.base import _validate_schema_value


MECHANICAL_OUTPUT_SOURCES = frozenset(
    {
        "worker_id",
        "session_id",
        "objective",
        "status",
        "created_at",
        "started_at",
        "run_count",
    }
)


@dataclass(slots=True, frozen=True)
class CallerOutputSpec:
    schema: dict[str, Any]
    mechanical_fields: dict[str, str]
    semantic_schema: dict[str, Any]

    def payload(self) -> dict[str, Any]:
        return {
            "schema": deepcopy(self.schema),
            "mechanical_fields": dict(self.mechanical_fields),
        }


def prepare_caller_output_spec(
    schema: dict[str, Any] | None,
    mechanical_fields: Mapping[str, str] | None = None,
) -> CallerOutputSpec | None:
    if schema is None:
        if mechanical_fields:
            raise ValueError("mechanical_fields requires output_schema")
        return None
    assert_portable_json_schema(schema, schema_name="caller_output_schema")
    bindings = dict(mechanical_fields or {})
    properties = schema["properties"]
    unknown_fields = set(bindings) - set(properties)
    if unknown_fields:
        raise ValueError(
            "mechanical_fields names are absent from output_schema: "
            + ", ".join(sorted(unknown_fields))
        )
    unknown_sources = set(bindings.values()) - MECHANICAL_OUTPUT_SOURCES
    if unknown_sources:
        raise ValueError(
            "unsupported mechanical field sources: "
            + ", ".join(sorted(unknown_sources))
        )
    semantic_properties = {
        name: deepcopy(property_schema)
        for name, property_schema in properties.items()
        if name not in bindings
    }
    semantic_schema = {
        "type": "object",
        "properties": semantic_properties,
        "required": list(semantic_properties),
        "additionalProperties": False,
    }
    assert_portable_json_schema(
        semantic_schema, schema_name="caller_semantic_output_schema"
    )
    return CallerOutputSpec(
        schema=deepcopy(schema),
        mechanical_fields=bindings,
        semantic_schema=semantic_schema,
    )


def merge_caller_output(
    spec: CallerOutputSpec,
    semantic_output: dict[str, Any],
    mechanical_values: Mapping[str, Any],
) -> dict[str, Any]:
    merged = dict(semantic_output)
    for output_field, source in spec.mechanical_fields.items():
        if source not in mechanical_values:
            raise ValueError(f"mechanical output source is unavailable: {source}")
        merged[output_field] = mechanical_values[source]
    _validate_schema_value(merged, spec.schema, path="caller_output")
    return merged
