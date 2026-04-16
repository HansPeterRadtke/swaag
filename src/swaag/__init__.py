"""swaag package."""

from __future__ import annotations

from swaag.config import AgentConfig, load_config
from swaag.runtime import AgentRuntime
from swaag.tests import main as _run_tests

__all__ = ["AgentConfig", "AgentRuntime", "load_config", "test"]
__version__ = "0.1.0"


def test(verbose: bool = False) -> int:
    """Run the packaged SWAAG smoke tests or the full repo tests when available."""
    return _run_tests(verbose=verbose)
