from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


_STORAGE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def is_storage_identifier(value: object) -> bool:
    return isinstance(value, str) and bool(_STORAGE_IDENTIFIER.fullmatch(value))


def validate_storage_identifier(value: object, *, label: str) -> str:
    if not is_storage_identifier(value):
        raise ValueError(
            f"{label} must be 1-128 ASCII letters, digits, dots, underscores, or hyphens and start with a letter or digit"
        )
    return str(value)


def scoped_storage_path(root: Path, identifier: object, *, label: str) -> Path:
    safe_identifier = validate_storage_identifier(identifier, label=label)
    resolved_root = Path(root).expanduser().resolve()
    candidate = (resolved_root / safe_identifier).resolve()
    if not candidate.is_relative_to(resolved_root):
        raise ValueError(f"{label} resolves outside its storage root")
    return candidate


def stable_json_dumps(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":") if indent is None else None, indent=indent)


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {k: to_jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
def expand_env_in_value(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [expand_env_in_value(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_env_in_value(item) for key, item in value.items()}
    return value
