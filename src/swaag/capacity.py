"""Mechanical admission limits; rejected work is never silently discarded."""
from __future__ import annotations


class QueueCapacityError(ValueError):
    pass


def positive_capacity(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def require_capacity(connection, query: str, parameters: tuple, *, limit: int, name: str) -> None:
    """Call inside the same BEGIN IMMEDIATE transaction as the admission write."""
    count = int(connection.execute(query, parameters).fetchone()[0])
    if count >= limit:
        raise QueueCapacityError(f"{name} capacity reached ({count}/{limit}); request rejected without discarding existing work")
