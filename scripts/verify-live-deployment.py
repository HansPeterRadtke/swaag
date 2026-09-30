#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import stat
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _run(*args: str, check: bool = True) -> str:
    completed = subprocess.run(
        list(args),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(args)}: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    return completed.stdout.strip()


def _systemctl_show(unit: str) -> dict[str, str]:
    fields = [
        "ActiveState",
        "SubState",
        "Result",
        "NRestarts",
        "MainPID",
        "User",
        "Group",
        "ExecStart",
        "FragmentPath",
    ]
    raw = _run("systemctl", "show", unit, "--property=" + ",".join(fields))
    result: dict[str, str] = {}
    for line in raw.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


def _url_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=3) as response:
        payload = response.read()
    decoded = json.loads(payload)
    return decoded if isinstance(decoded, dict) else {"value": decoded}


def _file_info(path: Path) -> dict[str, Any]:
    st = path.stat()
    return {
        "path": str(path),
        "uid": st.st_uid,
        "gid": st.st_gid,
        "mode": oct(stat.S_IMODE(st.st_mode)),
        "bytes": st.st_size,
        "sha256": _sha256(path),
        "mtime_ns": st.st_mtime_ns,
    }


def _git_state(repo: Path) -> dict[str, Any]:
    head = _run("git", "-C", str(repo), "rev-parse", "HEAD")
    status = _run("git", "-C", str(repo), "status", "--short", "--branch")
    diff = _run("git", "-C", str(repo), "diff", "--binary", "--no-ext-diff", "--", ".")
    untracked = _run(
        "git", "-C", str(repo), "ls-files", "--others", "--exclude-standard"
    ).splitlines()
    h = hashlib.sha256()
    h.update(head.encode())
    h.update(b"\0")
    h.update(diff.encode())
    for relative in sorted(x for x in untracked if x):
        path = repo / relative
        if not path.is_file():
            continue
        h.update(relative.encode())
        h.update(b"\0")
        h.update(path.read_bytes())
    return {
        "head": head,
        "status": status,
        "dirty_identity_sha256": h.hexdigest(),
        "untracked_files": sorted(x for x in untracked if x),
    }


def _package_manifest(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if path.suffix in {".pyc", ".pyo"}:
            continue
        result[path.relative_to(root).as_posix()] = _sha256(path)
    return result


def _operations_evidence(path: Path) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    if path.exists():
        for line in path.read_text(errors="replace").splitlines()[-500:]:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                events.append(item)
    names = [str(item.get("event", "")) for item in events]
    return {
        "path": str(path),
        "bytes": path.stat().st_size if path.exists() else 0,
        "recent_event_names": names,
        "has_startup": "startup" in names,
        "has_ready": "communication_ready" in names,
        "has_stopping": "communication_stopping" in names,
        "has_shutdown": "shutdown" in names,
        "recent_queue_overflow_count": sum(name == "queue_overflow" for name in names),
    }


def _sqlite_user_version(path: Path) -> int | None:
    if not path.exists():
        return None
    with sqlite3.connect(path) as connection:
        return int(connection.execute("PRAGMA user_version").fetchone()[0])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify the deployed SWAAG daemon against the tested repo/Infra state."
    )
    parser.add_argument("--repo", default="/data/src/github/swaag")
    parser.add_argument("--infra", default="/data/infra")
    parser.add_argument("--venv", default="/data/var/swaag/venv")
    parser.add_argument("--runtime-root", default="/data/var/swaag")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    infra = Path(args.infra).resolve()
    venv = Path(args.venv).resolve()
    runtime_root = Path(args.runtime_root).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    checks: list[dict[str, Any]] = []

    def check(check_name: str, passed: bool, **evidence: Any) -> None:
        checks.append({"name": check_name, "passed": bool(passed), "evidence": evidence})

    units = ["swaag-communication.service", "swaag-otel-collector.service"]
    unit_evidence: dict[str, Any] = {}
    for unit in units:
        source = infra / "etc/systemd/system" / unit
        installed = Path("/etc/systemd/system") / unit
        info = {
            "source": _file_info(source),
            "installed": _file_info(installed),
            "systemctl": _systemctl_show(unit),
        }
        unit_evidence[unit] = info
        check(
            f"unit:{unit}:source_matches_installed",
            info["source"]["sha256"] == info["installed"]["sha256"],
            source_sha256=info["source"]["sha256"],
            installed_sha256=info["installed"]["sha256"],
        )
        check(
            f"unit:{unit}:root_owned_0644",
            info["installed"]["uid"] == 0
            and info["installed"]["gid"] == 0
            and info["installed"]["mode"] == "0o644",
            installed=info["installed"],
        )
        show = info["systemctl"]
        check(
            f"service:{unit}:active",
            show.get("ActiveState") == "active" and show.get("SubState") == "running",
            systemctl=show,
        )
        check(
            f"service:{unit}:zero_restarts",
            int(show.get("NRestarts", "-1") or -1) == 0,
            systemctl=show,
        )

    wakeup = _systemctl_show("swaag-wakeup-dispatcher.service")
    check(
        "service:wakeup_dispatcher:displaced",
        wakeup.get("ActiveState") != "active",
        systemctl=wakeup,
    )

    package_root_raw = _run(
        str(venv / "bin/python"),
        "-c",
        "import pathlib,swaag; print(pathlib.Path(swaag.__file__).resolve().parent)",
    )
    package_root = Path(package_root_raw)
    repo_manifest = _package_manifest(repo / "src/swaag")
    deployed_manifest = _package_manifest(package_root)
    manifest_match = repo_manifest == deployed_manifest
    check(
        "package:repo_matches_deployed",
        manifest_match,
        repo_file_count=len(repo_manifest),
        deployed_file_count=len(deployed_manifest),
        repo_only=sorted(set(repo_manifest) - set(deployed_manifest))[:50],
        deployed_only=sorted(set(deployed_manifest) - set(repo_manifest))[:50],
        differing=[
            key
            for key in sorted(set(repo_manifest) & set(deployed_manifest))
            if repo_manifest[key] != deployed_manifest[key]
        ][:50],
    )

    agent_card = _url_json("http://127.0.0.1:13401/.well-known/agent-card.json")
    collector_health = _url_json("http://127.0.0.1:13502/")
    interfaces = agent_card.get("supportedInterfaces")
    bindings = {
        (str(item.get("protocolBinding")), str(item.get("protocolVersion")))
        for item in interfaces
        if isinstance(item, dict)
    } if isinstance(interfaces, list) else set()
    check(
        "communication:agent_card_reachable",
        bool(agent_card.get("name"))
        and {("JSONRPC", "1.0"), ("HTTP+JSON", "1.0")} <= bindings,
        agent_name=agent_card.get("name"),
        supportedInterfaces=interfaces,
        capabilities=agent_card.get("capabilities"),
    )
    check(
        "otel:collector_health",
        str(collector_health.get("status", "")).lower() in {"server available", "ok", "ready"},
        response=collector_health,
    )

    telemetry: dict[str, Any] = {}
    for name in ("traces.json", "metrics.json"):
        path = runtime_root / "telemetry" / name
        telemetry[name] = _file_info(path) if path.exists() else {"path": str(path), "missing": True}
        check(
            f"otel:{name}:nonempty",
            path.exists() and path.stat().st_size > 0,
            **telemetry[name],
        )

    operations = _operations_evidence(runtime_root / "logs/operations.jsonl")
    check(
        "operations:lifecycle_evidence",
        operations["has_startup"]
        and operations["has_ready"]
        and operations["has_stopping"]
        and operations["has_shutdown"],
        **operations,
    )

    communication_db = runtime_root / "sessions/communication.sqlite3"
    schema_version = _sqlite_user_version(communication_db)
    check(
        "runtime:communication_schema_current",
        schema_version == 6,
        path=str(communication_db),
        user_version=schema_version,
    )

    git = _git_state(repo)
    requirements = runtime_root / "production-requirements.lock.txt"
    report = {
        "benchmark": "live-deployment-verifier",
        "generated_at": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        "repo": str(repo),
        "infra": str(infra),
        "venv": str(venv),
        "runtime_root": str(runtime_root),
        "git": git,
        "production_requirements": _file_info(requirements)
        if requirements.exists()
        else {"path": str(requirements), "missing": True},
        "units": unit_evidence,
        "wakeup_dispatcher": wakeup,
        "package_root": str(package_root),
        "telemetry": telemetry,
        "operations": operations,
        "communication_schema_user_version": schema_version,
        "checks": checks,
        "passed": all(item["passed"] for item in checks),
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "checks": len(checks), "output": str(output)}, sort_keys=True))
    if not report["passed"]:
        for item in checks:
            if not item["passed"]:
                print(json.dumps(item, sort_keys=True), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
