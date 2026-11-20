#!/usr/bin/env python3
"""Fail-closed environment preflight for SWAAG real read-only agent demo.

This script inspects the ACTIVE interpreter and exact source. It cannot
install packages, upgrade SQLite, contact an LLM or change service state.
"""
from __future__ import annotations

import argparse
import ast
import importlib
import json
from pathlib import Path
import sqlite3
import sys


REQUIRED_MODULES=(
    'pytest',
    'opentelemetry.context',
    'opentelemetry.metrics',
    'opentelemetry.trace',
    'opentelemetry.sdk',
    'opentelemetry.exporter.otlp.proto.http',
)


def classify_env(*,python: tuple[int,int],sqlite: tuple[int,int,int],
                 grammar_ok: bool,available_modules: dict[str,bool]) -> dict:
    problems=[]
    if python < (3,11):
        problems.append('unsupported_python_version')
    if not grammar_ok:
        problems.append('syntax_incompatible_with_claimed_python311')
    for name in REQUIRED_MODULES:
        if available_modules.get(name) is not True:
            problems.append('missing_dependency:'+name)
    if sqlite < (3,51,3):
        problems.append('sqlite_runtime_older_than_required_3_51_3')
    return {'status':'ready_for_deterministic_tests' if not problems
            else 'environment_blocked',
            'problems':problems,
            'cannot_infer_live_agent_success':True}


def inspect(repo:Path) -> dict:
    file=repo/'src/swaag/foreground_fast.py'
    source=file.read_text(encoding='utf8')
    grammar_error=''
    try:ast.parse(source,filename=str(file),feature_version=(3,11))
    except SyntaxError as exc:grammar_error=str(exc)
    modules={}
    for name in REQUIRED_MODULES:
        try:
            importlib.import_module(name)
            modules[name]=True
        except (ImportError,ModuleNotFoundError,AttributeError):
            modules[name]=False
    parts=tuple(int(x) for x in sqlite3.sqlite_version.split('.'))
    result=classify_env(python=sys.version_info[:2],sqlite=parts,
                        grammar_ok=not grammar_error,
                        available_modules=modules)
    result.update({'python':sys.version.split()[0],
                   'sqlite_version':sqlite3.sqlite_version,
                   'foreground_fast_grammar_error':grammar_error,
                   'dependent_module_presence':modules})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    value=inspect(args.repo)
    if args.output:
        if args.output.exists():parser.error('Evidence exists: preserve it')
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(value,indent=2)+'\n')
    print('SWAAG_RUNTIME_PREFLIGHT',json.dumps(value,sort_keys=True))
    return 0 if value['status']=='ready_for_deterministic_tests' else 2


if __name__=='__main__':
    raise SystemExit(main())
