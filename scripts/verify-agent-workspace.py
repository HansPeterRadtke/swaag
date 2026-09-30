#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path
from tempfile import TemporaryDirectory

from swaag.config import load_config
from swaag.runtime import AgentRuntime
from swaag.tools.registry import ToolRegistry
from swaag.types import SessionState


def _state(config) -> SessionState:
    runtime = AgentRuntime(config, model_client=object())
    return runtime.create_or_load_session()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--root", default="/data/var/swaag-benchmarks/agent-workspace-live/root")
    args = parser.parse_args()
    out = Path(args.output).resolve()
    root = Path(args.root).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    project_probe = out.parent / "project-visible-only.txt"
    project_probe.write_text("project-secret", encoding="utf-8")
    config = load_config(
        env={
            "SWAAG__SESSIONS__ROOT": str(out.parent / "sessions"),
            "SWAAG__AGENT_DATA__ROOT": str(root),
            "SWAAG__TOOLS__ALLOW_STATEFUL_TOOLS": "true",
            "SWAAG__TOOLS__ALLOW_SIDE_EFFECT_TOOLS": "false",
        }
    )
    registry = ToolRegistry()
    state = _state(config)
    base = {
        "code": None,
        "command": None,
        "path": None,
        "text": None,
        "packages": None,
        "max_chars": 8000,
    }
    checks: list[dict] = []

    def check(name: str, passed: bool, **evidence):
        checks.append({"name": name, "passed": bool(passed), "evidence": evidence})

    _, result = registry.dispatch(
        "agent_workspace",
        {
            **base,
            "operation": "shell",
            "command": (
                "pwd; test ! -e /etc/passwd; "
                f"test ! -e '{project_probe}'; "
                "printf live > retained.txt"
            ),
        },
        config,
        state,
    )
    check(
        "sandbox_filesystem_isolation",
        result.output["return_code"] == 0
        and result.output["stdout"].splitlines()[0] == "/agent/workspace",
        result=result.output,
    )
    retained = root / "workspace/retained.txt"
    check(
        "workspace_persistence",
        retained.exists() and retained.read_text() == "live",
        path=str(retained),
    )

    _, py = registry.dispatch(
        "agent_workspace",
        {
            **base,
            "operation": "python",
            "code": "from pathlib import Path; print(Path('/etc/passwd').exists()); print(Path('retained.txt').read_text())",
        },
        config,
        state,
    )
    check(
        "python_private_venv_and_isolation",
        py.output["return_code"] == 0
        and py.output["stdout"].splitlines() == ["False", "live"]
        and (root / "python-venv/bin/python").exists(),
        result=py.output,
    )

    _, net = registry.dispatch(
        "agent_workspace",
        {
            **base,
            "operation": "python",
            "code": (
                "import socket\n"
                "s=socket.socket(); s.settimeout(1)\n"
                "try:\n s.connect(('1.1.1.1',53)); print('network-visible')\n"
                "except Exception as e:\n print(type(e).__name__)\n"
            ),
        },
        config,
        state,
    )
    check(
        "ordinary_python_has_no_network",
        net.output["return_code"] == 0 and "network-visible" not in net.output["stdout"],
        result=net.output,
    )

    side_effect_blocked = False
    error = ""
    try:
        registry.dispatch(
            "agent_workspace",
            {
                **base,
                "operation": "pip_install",
                "packages": ["this-package-should-never-be-installed"],
            },
            config,
            state,
        )
    except Exception as exc:
        side_effect_blocked = True
        error = f"{type(exc).__name__}: {exc}"
    check("package_install_side_effect_gated", side_effect_blocked, error=error)

    traversal_blocked = False
    error = ""
    try:
        registry.dispatch(
            "agent_workspace",
            {**base, "operation": "read_text", "path": "../escape.txt"},
            config,
            state,
        )
    except Exception as exc:
        traversal_blocked = True
        error = f"{type(exc).__name__}: {exc}"
    check("path_traversal_blocked", traversal_blocked, error=error)

    report = {
        "benchmark": "agent-workspace-live-verifier",
        "root": str(root),
        "checks": checks,
        "passed": all(item["passed"] for item in checks),
    }
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "checks": len(checks), "output": str(out)}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
