from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from swaag.schema_portability import assert_portable_json_schema
from swaag.sqlite_schema import apply_sqlite_migrations
from swaag.tools.base import _validate_schema_value
from swaag.utils import new_id, sha256_text, stable_json_dumps, utc_now_iso


_DELEGATED_TOOL_STORE_MIGRATIONS = (
    (
        """
        CREATE TABLE delegated_tool_catalogs (
            session_id TEXT NOT NULL,
            revision INTEGER NOT NULL,
            source TEXT NOT NULL,
            external_context_id TEXT NOT NULL,
            external_request_id TEXT NOT NULL,
            tools_json TEXT NOT NULL,
            catalog_sha256 TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (session_id, revision),
            UNIQUE (source, external_request_id)
        )
        """,
        """
        CREATE INDEX delegated_tool_catalogs_latest
        ON delegated_tool_catalogs(session_id, revision DESC)
        """,
        """
        CREATE TABLE delegated_tool_calls (
            call_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            catalog_revision INTEGER NOT NULL,
            tool_name TEXT NOT NULL,
            arguments_json TEXT NOT NULL,
            arguments_sha256 TEXT NOT NULL,
            status TEXT NOT NULL,
            requested_at TEXT NOT NULL,
            result_source TEXT,
            result_external_request_id TEXT,
            result_message_id TEXT,
            result_content TEXT,
            result_error TEXT,
            result_metadata_json TEXT,
            resolved_at TEXT,
            history_event_type TEXT,
            history_event_sequence INTEGER,
            history_event_hash TEXT,
            FOREIGN KEY (session_id, catalog_revision)
                REFERENCES delegated_tool_catalogs(session_id, revision),
            UNIQUE (result_source, result_message_id)
        )
        """,
        """
        CREATE UNIQUE INDEX delegated_tool_calls_one_pending
        ON delegated_tool_calls(session_id) WHERE status='pending'
        """,
        """
        CREATE INDEX delegated_tool_calls_session
        ON delegated_tool_calls(session_id, requested_at, call_id)
        """,
    ),
)


@dataclass(slots=True, frozen=True)
class DelegatedToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    metadata: dict[str, Any]

    def prompt_tuple(self) -> tuple[str, str, dict[str, Any], str]:
        if self.metadata.get("external_execution_mode") == "runtime":
            guidance = (
                "This is an external tool that SWAAG can execute through its configured provider "
                "adapter. If selected, the exact provider result returns to the current run."
            )
        else:
            guidance = (
                "Execution is delegated to the connected external tool provider. If selected, "
                "the current run pauses until that provider returns the exact result."
            )
        return self.name, self.description, self.parameters, guidance


@dataclass(slots=True, frozen=True)
class DelegatedToolResultInput:
    message_id: str
    call_id: str
    content: str
    error: str | None
    metadata: dict[str, Any]


@dataclass(slots=True, frozen=True)
class DelegatedToolCatalog:
    session_id: str
    revision: int
    source: str
    external_context_id: str
    external_request_id: str
    tools: tuple[DelegatedToolSpec, ...]
    catalog_sha256: str
    created_at: str


@dataclass(slots=True, frozen=True)
class DelegatedToolCall:
    call_id: str
    session_id: str
    catalog_revision: int
    tool_name: str
    arguments: dict[str, Any]
    arguments_sha256: str
    status: str
    requested_at: str
    result_source: str | None
    result_external_request_id: str | None
    result_message_id: str | None
    result_content: str | None
    result_error: str | None
    result_metadata: dict[str, Any] | None
    resolved_at: str | None
    history_event_type: str | None
    history_event_sequence: int | None
    history_event_hash: str | None


# Public architecture terminology: delegated tools are layer-3 external tools.
# Compatibility names remain because protocol adapters already persist them.
ExternalToolSpec = DelegatedToolSpec
ExternalToolCatalog = DelegatedToolCatalog
ExternalToolCall = DelegatedToolCall


class DelegatedToolInputRequired(RuntimeError):
    def __init__(self, call: DelegatedToolCall):
        super().__init__(
            f"Delegated tool {call.tool_name} is waiting for client result "
            f"{call.call_id}"
        )
        self.call = call


def prepare_delegated_tool_spec(payload: dict[str, Any]) -> DelegatedToolSpec:
    if not isinstance(payload, dict):
        raise ValueError("delegated tool definition must be an object")
    raw_name = payload.get("name")
    if not isinstance(raw_name, str) or not raw_name.strip():
        raise ValueError("delegated tool name must be a non-empty string")
    name = raw_name.strip()
    if name != raw_name:
        raise ValueError("delegated tool name must not contain surrounding whitespace")
    if len(name) > 128:
        raise ValueError("delegated tool name exceeds 128 characters")
    description = payload.get("description")
    if not isinstance(description, str):
        raise ValueError(f"delegated tool {name} description must be a string")
    if len(description) > 8192:
        raise ValueError(f"delegated tool {name} description is too large")
    parameters = payload.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError(f"delegated tool {name} parameters must be a schema object")
    assert_portable_json_schema(parameters, schema_name=f"delegated tool {name}")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"delegated tool {name} metadata must be an object")

    # Round-trip through canonical JSON so callers cannot mutate retained input.
    normalized = json.loads(
        stable_json_dumps(
            {
                "parameters": parameters,
                "metadata": metadata,
            },
            indent=None,
        )
    )
    return DelegatedToolSpec(
        name=name,
        description=description,
        parameters=normalized["parameters"],
        metadata=normalized["metadata"],
    )


class DelegatedToolStore:
    """Durable client-executed tool catalogs, calls, and exact result lineage."""

    def __init__(self, root: Path):
        self.path = Path(root).expanduser() / "delegated_tools.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            apply_sqlite_migrations(
                connection,
                store_name="delegated tool store",
                migrations=_DELEGATED_TOOL_STORE_MIGRATIONS,
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    @staticmethod
    def _spec_payload(spec: DelegatedToolSpec) -> dict[str, Any]:
        return {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.parameters,
            "metadata": spec.metadata,
        }

    @classmethod
    def _catalog(cls, row: sqlite3.Row) -> DelegatedToolCatalog:
        tools_json = str(row["tools_json"])
        if sha256_text(tools_json) != str(row["catalog_sha256"]):
            raise RuntimeError("delegated tool catalog hash verification failed")
        raw_tools = json.loads(tools_json)
        if not isinstance(raw_tools, list):
            raise RuntimeError("delegated tool catalog contains invalid JSON")
        tools = tuple(prepare_delegated_tool_spec(item) for item in raw_tools)
        return DelegatedToolCatalog(
            session_id=str(row["session_id"]),
            revision=int(row["revision"]),
            source=str(row["source"]),
            external_context_id=str(row["external_context_id"]),
            external_request_id=str(row["external_request_id"]),
            tools=tools,
            catalog_sha256=str(row["catalog_sha256"]),
            created_at=str(row["created_at"]),
        )

    @staticmethod
    def _call(row: sqlite3.Row) -> DelegatedToolCall:
        arguments_json = str(row["arguments_json"])
        if sha256_text(arguments_json) != str(row["arguments_sha256"]):
            raise RuntimeError("delegated tool call argument hash verification failed")
        arguments = json.loads(arguments_json)
        if not isinstance(arguments, dict):
            raise RuntimeError("delegated tool call arguments are not an object")
        status = str(row["status"])
        if status not in {"pending", "resolved", "failed", "canceled"}:
            raise RuntimeError(f"delegated tool call has invalid status: {status}")
        metadata_json = row["result_metadata_json"]
        result_metadata = None if metadata_json is None else json.loads(str(metadata_json))
        if result_metadata is not None and not isinstance(result_metadata, dict):
            raise RuntimeError("delegated tool result metadata is not an object")
        return DelegatedToolCall(
            call_id=str(row["call_id"]),
            session_id=str(row["session_id"]),
            catalog_revision=int(row["catalog_revision"]),
            tool_name=str(row["tool_name"]),
            arguments=arguments,
            arguments_sha256=str(row["arguments_sha256"]),
            status=status,
            requested_at=str(row["requested_at"]),
            result_source=(
                None if row["result_source"] is None else str(row["result_source"])
            ),
            result_external_request_id=(
                None
                if row["result_external_request_id"] is None
                else str(row["result_external_request_id"])
            ),
            result_message_id=(
                None
                if row["result_message_id"] is None
                else str(row["result_message_id"])
            ),
            result_content=(
                None
                if row["result_content"] is None
                else str(row["result_content"])
            ),
            result_error=(
                None if row["result_error"] is None else str(row["result_error"])
            ),
            result_metadata=result_metadata,
            resolved_at=(
                None if row["resolved_at"] is None else str(row["resolved_at"])
            ),
            history_event_type=(
                None
                if row["history_event_type"] is None
                else str(row["history_event_type"])
            ),
            history_event_sequence=(
                None
                if row["history_event_sequence"] is None
                else int(row["history_event_sequence"])
            ),
            history_event_hash=(
                None
                if row["history_event_hash"] is None
                else str(row["history_event_hash"])
            ),
        )

    def bind_catalog(
        self,
        session_id: str,
        *,
        source: str,
        external_context_id: str,
        external_request_id: str,
        tools: Iterable[DelegatedToolSpec],
    ) -> DelegatedToolCatalog:
        catalog_tools = tuple(
            prepare_delegated_tool_spec(self._spec_payload(tool)) for tool in tools
        )
        names = [tool.name for tool in catalog_tools]
        if len(names) != len(set(names)):
            raise ValueError("delegated tool names must be unique within a catalog")
        tools_json = stable_json_dumps(
            [self._spec_payload(tool) for tool in catalog_tools], indent=None
        )
        catalog_sha256 = sha256_text(tools_json)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """
                SELECT * FROM delegated_tool_catalogs
                WHERE source=? AND external_request_id=?
                """,
                (source, external_request_id),
            ).fetchone()
            if existing is not None:
                if (
                    str(existing["session_id"]) != session_id
                    or str(existing["external_context_id"]) != external_context_id
                ):
                    raise ValueError(
                        "delegated tool request is already bound to another context"
                    )
                if (
                    str(existing["catalog_sha256"]) != catalog_sha256
                    or str(existing["tools_json"]) != tools_json
                ):
                    raise ValueError(
                        "delegated tool request is already bound to a different "
                        "exact catalog"
                    )
                return self._catalog(existing)
            row = connection.execute(
                """
                SELECT COALESCE(MAX(revision), 0)
                FROM delegated_tool_catalogs WHERE session_id=?
                """,
                (session_id,),
            ).fetchone()
            revision = int(row[0]) + 1
            connection.execute(
                """
                INSERT INTO delegated_tool_catalogs(
                    session_id, revision, source, external_context_id,
                    external_request_id, tools_json, catalog_sha256, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    revision,
                    source,
                    external_context_id,
                    external_request_id,
                    tools_json,
                    catalog_sha256,
                    utc_now_iso(),
                ),
            )
            stored = connection.execute(
                """
                SELECT * FROM delegated_tool_catalogs
                WHERE session_id=? AND revision=?
                """,
                (session_id, revision),
            ).fetchone()
        if stored is None:
            raise RuntimeError("delegated tool catalog was not stored")
        return self._catalog(stored)

    def latest_catalog(self, session_id: str) -> DelegatedToolCatalog | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM delegated_tool_catalogs
                WHERE session_id=? ORDER BY revision DESC LIMIT 1
                """,
                (session_id,),
            ).fetchone()
        return None if row is None else self._catalog(row)

    def catalog(
        self, session_id: str, revision: int
    ) -> DelegatedToolCatalog | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM delegated_tool_catalogs
                WHERE session_id=? AND revision=?
                """,
                (session_id, int(revision)),
            ).fetchone()
        return None if row is None else self._catalog(row)

    def request_call(
        self,
        session_id: str,
        *,
        catalog_revision: int,
        tool_name: str,
        arguments: dict[str, Any],
        call_id: str | None = None,
    ) -> DelegatedToolCall:
        arguments_json = stable_json_dumps(arguments, indent=None)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            catalog_row = connection.execute(
                """
                SELECT * FROM delegated_tool_catalogs
                WHERE session_id=? AND revision=?
                """,
                (session_id, int(catalog_revision)),
            ).fetchone()
            if catalog_row is None:
                raise ValueError("delegated tool catalog revision is unavailable")
            catalog = self._catalog(catalog_row)
            spec = next((item for item in catalog.tools if item.name == tool_name), None)
            if spec is None:
                raise ValueError(f"delegated tool is not in catalog: {tool_name}")
            _validate_schema_value(
                arguments, spec.parameters, path=f"delegated tool {tool_name}"
            )
            pending = connection.execute(
                """
                SELECT call_id FROM delegated_tool_calls
                WHERE session_id=? AND status='pending'
                """,
                (session_id,),
            ).fetchone()
            if pending is not None:
                raise ValueError(
                    f"session already awaits delegated tool call {pending['call_id']}"
                )
            resolved_call_id = call_id or new_id("delegated_tool_call")
            connection.execute(
                """
                INSERT INTO delegated_tool_calls(
                    call_id, session_id, catalog_revision, tool_name,
                    arguments_json, arguments_sha256, status, requested_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)
                """,
                (
                    resolved_call_id,
                    session_id,
                    catalog.revision,
                    tool_name,
                    arguments_json,
                    sha256_text(arguments_json),
                    utc_now_iso(),
                ),
            )
            row = connection.execute(
                "SELECT * FROM delegated_tool_calls WHERE call_id=?",
                (resolved_call_id,),
            ).fetchone()
        if row is None:
            raise RuntimeError("delegated tool call was not stored")
        return self._call(row)

    def pending_call(self, session_id: str) -> DelegatedToolCall | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM delegated_tool_calls
                WHERE session_id=? AND status='pending'
                ORDER BY requested_at, call_id LIMIT 1
                """,
                (session_id,),
            ).fetchone()
        return None if row is None else self._call(row)

    def call(self, call_id: str) -> DelegatedToolCall | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM delegated_tool_calls WHERE call_id=?",
                (call_id,),
            ).fetchone()
        return None if row is None else self._call(row)

    def resolve_call(
        self,
        session_id: str,
        call_id: str,
        *,
        source: str,
        external_request_id: str,
        result: DelegatedToolResultInput,
    ) -> DelegatedToolCall:
        if result.call_id != call_id:
            raise ValueError("delegated tool result addresses a different call")
        if not isinstance(result.message_id, str) or not result.message_id:
            raise ValueError("delegated tool result message id must be non-empty")
        if not isinstance(result.content, str):
            raise ValueError("delegated tool result content must be a string")
        if result.error is not None and not isinstance(result.error, str):
            raise ValueError("delegated tool result error must be a string")
        if not isinstance(result.metadata, dict):
            raise ValueError("delegated tool result metadata must be an object")
        metadata_json = stable_json_dumps(result.metadata, indent=None)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM delegated_tool_calls WHERE call_id=?",
                (call_id,),
            ).fetchone()
            if row is None or str(row["session_id"]) != session_id:
                raise ValueError("delegated tool call is unknown for this session")
            current = self._call(row)
            terminal_values = (
                source,
                external_request_id,
                result.message_id,
                result.content,
                result.error,
                result.metadata,
            )
            if current.status in {"resolved", "failed"}:
                existing_values = (
                    current.result_source,
                    current.result_external_request_id,
                    current.result_message_id,
                    current.result_content,
                    current.result_error,
                    current.result_metadata,
                )
                if existing_values != terminal_values:
                    raise ValueError(
                        "delegated tool call already has a different exact result"
                    )
                return current
            if current.status != "pending":
                raise ValueError(
                    f"delegated tool call cannot resolve from {current.status}"
                )
            status = "failed" if result.error is not None else "resolved"
            connection.execute(
                """
                UPDATE delegated_tool_calls SET
                    status=?, result_source=?, result_external_request_id=?,
                    result_message_id=?, result_content=?, result_error=?,
                    result_metadata_json=?, resolved_at=?
                WHERE call_id=? AND status='pending'
                """,
                (
                    status,
                    source,
                    external_request_id,
                    result.message_id,
                    result.content,
                    result.error,
                    metadata_json,
                    utc_now_iso(),
                    call_id,
                ),
            )
            stored = connection.execute(
                "SELECT * FROM delegated_tool_calls WHERE call_id=?",
                (call_id,),
            ).fetchone()
        if stored is None:
            raise RuntimeError("delegated tool result was not stored")
        return self._call(stored)

    def verify_result_message(
        self,
        session_id: str,
        result: DelegatedToolResultInput,
    ) -> DelegatedToolCall:
        """Verify a repeated protocol message against one terminal exact result."""
        call = self.call(result.call_id)
        if call is None or call.session_id != session_id:
            raise ValueError("delegated tool result references an unknown call")
        if call.status not in {"resolved", "failed"}:
            raise ValueError(
                f"delegated tool call has no terminal result: {call.call_id}"
            )
        if (
            call.result_message_id,
            call.result_content,
            call.result_error,
            call.result_metadata,
        ) != (
            result.message_id,
            result.content,
            result.error,
            result.metadata,
        ):
            raise ValueError(
                "delegated tool result message differs from durable exact result"
            )
        return call

    def link_history(
        self,
        call_id: str,
        *,
        event_type: str,
        sequence: int,
        event_hash: str,
    ) -> DelegatedToolCall:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM delegated_tool_calls WHERE call_id=?",
                (call_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"unknown delegated tool call: {call_id}")
            current = self._call(row)
            existing = (
                current.history_event_type,
                current.history_event_sequence,
                current.history_event_hash,
            )
            requested = (event_type, int(sequence), event_hash)
            if current.history_event_sequence is not None and existing != requested:
                raise ValueError(
                    "delegated tool call is already linked to different history"
                )
            connection.execute(
                """
                UPDATE delegated_tool_calls SET
                    history_event_type=?, history_event_sequence=?, history_event_hash=?
                WHERE call_id=?
                """,
                (event_type, int(sequence), event_hash, call_id),
            )
            stored = connection.execute(
                "SELECT * FROM delegated_tool_calls WHERE call_id=?",
                (call_id,),
            ).fetchone()
        if stored is None:
            raise RuntimeError("delegated tool history link was not stored")
        return self._call(stored)

    def cancel_pending(self, session_id: str, *, reason: str) -> DelegatedToolCall | None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT * FROM delegated_tool_calls
                WHERE session_id=? AND status='pending'
                ORDER BY requested_at, call_id LIMIT 1
                """,
                (session_id,),
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                """
                UPDATE delegated_tool_calls SET
                    status='canceled', result_error=?, resolved_at=?
                WHERE call_id=? AND status='pending'
                """,
                (reason, utc_now_iso(), str(row["call_id"])),
            )
            stored = connection.execute(
                "SELECT * FROM delegated_tool_calls WHERE call_id=?",
                (str(row["call_id"]),),
            ).fetchone()
        return None if stored is None else self._call(stored)
