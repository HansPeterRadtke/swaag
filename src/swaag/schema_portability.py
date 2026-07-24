from __future__ import annotations

from typing import Any


class PortableSchemaError(ValueError):
    pass


_ALLOWED_KEYS = frozenset({"type", "properties", "required", "additionalProperties", "enum", "items", "anyOf"})
_FORBIDDEN_KEYS = frozenset(
    {
        "allOf",
        "oneOf",
        "if",
        "then",
        "else",
        "const",
        "patternProperties",
        "minProperties",
        "maxProperties",
        "minItems",
        "maxItems",
        "minLength",
        "maxLength",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "pattern",
        "format",
    }
)


def assert_portable_json_schema(schema: dict[str, Any], *, schema_name: str = "schema") -> None:
    _validate_schema(schema, path=schema_name, root=True)


def _validate_schema(schema: Any, *, path: str, root: bool = False) -> None:
    if not isinstance(schema, dict):
        raise PortableSchemaError(f"{path} must be a schema object")
    unknown = set(schema) - _ALLOWED_KEYS
    forbidden = set(schema) & _FORBIDDEN_KEYS
    if unknown or forbidden:
        keys = sorted(unknown | forbidden)
        raise PortableSchemaError(f"{path} uses nonportable schema keys: {', '.join(keys)}")

    if "anyOf" in schema:
        if root:
            raise PortableSchemaError(f"{path} must be a root object, not anyOf")
        variants = schema.get("anyOf")
        if not isinstance(variants, list) or not variants:
            raise PortableSchemaError(f"{path}.anyOf must be a non-empty list")
        for index, variant in enumerate(variants):
            _validate_schema(variant, path=f"{path}.anyOf[{index}]")
        return

    schema_type = schema.get("type")
    if root and schema_type != "object":
        raise PortableSchemaError(f"{path} root type must be object")
    if isinstance(schema_type, list):
        raise PortableSchemaError(f"{path}.type must be a single type string")
    if schema_type == "object":
        _validate_object_schema(schema, path=path)
        return
    if schema_type == "array":
        items = schema.get("items")
        if items is None:
            raise PortableSchemaError(f"{path}.items is required for arrays")
        _validate_schema(items, path=f"{path}.items")
        return
    if schema_type in {"string", "integer", "number", "boolean", "null"}:
        enum = schema.get("enum")
        if enum is not None and not isinstance(enum, list):
            raise PortableSchemaError(f"{path}.enum must be a list")
        return
    raise PortableSchemaError(f"{path}.type is unsupported or missing: {schema_type!r}")


def _validate_object_schema(schema: dict[str, Any], *, path: str) -> None:
    properties = schema.get("properties")
    required = schema.get("required")
    if not isinstance(properties, dict):
        raise PortableSchemaError(f"{path}.properties must be present for every object")
    if not isinstance(required, list):
        raise PortableSchemaError(f"{path}.required must be present for every object")
    if schema.get("additionalProperties") is not False:
        raise PortableSchemaError(f"{path}.additionalProperties must be false")
    property_keys = set(properties)
    required_keys = {str(item) for item in required}
    if property_keys != required_keys:
        missing = sorted(property_keys - required_keys)
        extra = sorted(required_keys - property_keys)
        details = []
        if missing:
            details.append(f"missing required keys: {', '.join(missing)}")
        if extra:
            details.append(f"required keys without properties: {', '.join(extra)}")
        raise PortableSchemaError(f"{path} must require every property ({'; '.join(details)})")
    for name, child in properties.items():
        _validate_schema(child, path=f"{path}.properties.{name}")
