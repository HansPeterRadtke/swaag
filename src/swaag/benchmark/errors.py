from __future__ import annotations


class LocalSwebenchFailure(RuntimeError):
    """Raised when the local SWE-bench harness cannot prepare or run a case."""

