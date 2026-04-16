from __future__ import annotations

import swaag
from swaag.cli import _build_parser


def test_import_package() -> None:
    assert hasattr(swaag, "AgentRuntime")
    assert hasattr(swaag, "load_config")


def test_cli_parser_exists() -> None:
    parser = _build_parser()
    assert parser.prog == "swaag"
