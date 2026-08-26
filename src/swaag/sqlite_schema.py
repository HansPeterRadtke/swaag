from __future__ import annotations

import sqlite3
from collections.abc import Sequence


class UnsupportedSchemaVersionError(RuntimeError):
    """Raised instead of opening durable state written by newer code."""


def apply_sqlite_migrations(
    connection: sqlite3.Connection,
    *,
    store_name: str,
    migrations: Sequence[Sequence[str]],
) -> int:
    """Apply ordered, idempotent migrations under one exclusive writer lock."""
    current = int(connection.execute("PRAGMA user_version").fetchone()[0])
    target = len(migrations)
    if current > target:
        raise UnsupportedSchemaVersionError(
            f"{store_name} schema version {current} is newer than supported version {target}"
        )
    if current == target:
        return current

    connection.execute("BEGIN IMMEDIATE")
    try:
        locked_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if locked_version > target:
            raise UnsupportedSchemaVersionError(
                f"{store_name} schema version {locked_version} is newer than "
                f"supported version {target}"
            )
        for version in range(locked_version + 1, target + 1):
            for statement in migrations[version - 1]:
                connection.execute(statement)
            connection.execute(f"PRAGMA user_version={version}")
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    return target
