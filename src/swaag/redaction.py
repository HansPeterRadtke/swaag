from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Iterable, Mapping
from typing import Any

_REDACTED = "[REDACTED]"
_SENSITIVE_KEYS = frozenset(
    {
        "authorization",
        "proxy_authorization",
        "api_key",
        "apikey",
        "access_token",
        "refresh_token",
        "bearer_token",
        "password",
        "passwd",
        "client_secret",
        "private_key",
        "credential",
        "credentials",
        "cookie",
        "set_cookie",
    }
)
_SENSITIVE_SUFFIXES = (
    "_api_key",
    "_access_token",
    "_refresh_token",
    "_bearer_token",
    "_password",
    "_passwd",
    "_client_secret",
    "_private_key",
    "_credential",
    "_credentials",
)
_AUTH_RE = re.compile(r"(?i)\b(authorization\s*[:=]\s*)?(bearer|basic)\s+([^\s,;]+)")
_KEY_VALUE_RE = re.compile(
    r"(?i)(\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|bearer[_-]?token|"
    r"password|passwd|client[_-]?secret|private[_-]?key|authorization|credential)s?\b"
    r"\s*[:=]\s*)([^\s,;\]}]+)"
)
_URL_CREDENTIAL_RE = re.compile(r"(https?://[^:/\s]+:)([^@/\s]+)(@)", re.IGNORECASE)
_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN ([A-Z0-9 ]*PRIVATE KEY)-----.*?-----END \1-----",
    re.DOTALL,
)


def _normalized_key(key: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(key).strip().casefold()).strip("_")


def is_sensitive_key(key: object) -> bool:
    normalized = _normalized_key(key)
    return normalized in _SENSITIVE_KEYS or any(
        normalized.endswith(suffix) for suffix in _SENSITIVE_SUFFIXES
    )


def _fingerprint(value: object) -> str:
    raw = str(value)
    digest = hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()[:12]
    return f"{_REDACTED}:sha256={digest}:chars={len(raw)}"


def redact_text(text: str, *, secret_values: Iterable[str] = ()) -> str:
    result = str(text)
    for raw_secret in secret_values:
        secret = str(raw_secret)
        if secret:
            result = result.replace(secret, _fingerprint(secret))
    result = _PRIVATE_KEY_RE.sub(lambda match: _fingerprint(match.group(0)), result)
    result = _AUTH_RE.sub(
        lambda match: f"{match.group(1) or ''}{match.group(2)} {_fingerprint(match.group(3))}",
        result,
    )
    result = _KEY_VALUE_RE.sub(
        lambda match: f"{match.group(1)}{_fingerprint(match.group(2))}", result
    )
    result = _URL_CREDENTIAL_RE.sub(
        lambda match: f"{match.group(1)}{_fingerprint(match.group(2))}{match.group(3)}",
        result,
    )
    return result


def redact_for_persistence(value: Any, *, secret_values: Iterable[str] = ()) -> Any:
    secrets = tuple(str(item) for item in secret_values if str(item))
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if is_sensitive_key(key_text):
                result[key_text] = _fingerprint(item)
            else:
                result[key_text] = redact_for_persistence(item, secret_values=secrets)
        return result
    if isinstance(value, list):
        return [redact_for_persistence(item, secret_values=secrets) for item in value]
    if isinstance(value, tuple):
        return [redact_for_persistence(item, secret_values=secrets) for item in value]
    if isinstance(value, str):
        return redact_text(value, secret_values=secrets)
    return value


def configured_secret_values(config: object) -> tuple[str, ...]:
    """Configured credential values that must not reach diagnostic persistence."""
    values: list[str] = []
    mcp = getattr(config, "mcp", None)
    auth = getattr(mcp, "authorization", None)
    model = getattr(config, "model", None)
    candidates = [
        getattr(auth, "introspection_client_secret", ""),
    ]
    env_name = str(getattr(model, "api_key_env", "") or "").strip()
    if env_name:
        candidates.append(os.environ.get(env_name, ""))
    for value in candidates:
        text = str(value or "")
        if text:
            values.append(text)
    return tuple(dict.fromkeys(values))
