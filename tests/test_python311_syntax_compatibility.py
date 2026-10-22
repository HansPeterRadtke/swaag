"""SWAAG declares Python >=3.11, so release sources must parse on 3.11.

The 2026-10-22 Nitro preflight discovered a Python 3.12-only multiline
expression inside an f-string on the foreground fast path. This independent
source test prevents accidentally raising the Python version floor by syntax.
"""
from __future__ import annotations

import ast
from pathlib import Path


def test_all_shipped_python_source_parses_with_supported_python_311_grammar():
    root = Path(__file__).resolve().parents[1]
    source_paths = [
        *root.joinpath('src', 'swaag').rglob('*.py'),
        *root.joinpath('scripts').rglob('*.py'),
    ]
    assert len(source_paths) >= 175
    failures = []
    for path in source_paths:
        try:
            ast.parse(path.read_text(encoding='utf-8-sig'),
                      filename=str(path), feature_version=(3, 11))
        except SyntaxError as exc:
            failures.append(f'{path.relative_to(root)}:{exc.lineno}: {exc.msg}')
    assert not failures, '\n'.join(failures)
