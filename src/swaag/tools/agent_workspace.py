from __future__ import annotations

import os
import fcntl
import heapq
import tempfile
import time
from contextlib import contextmanager
import shutil
from pathlib import Path
from typing import Any

from swaag.tools.base import Tool, ToolContext, ToolExecutionError, ToolValidationError
from swaag.environment.artifacts import TextArtifactStore
from swaag.environment.capture import capture_command
from swaag.types import ToolExecutionResult, ToolGeneratedEvent, ToolKind
from swaag.utils import stable_json_dumps, sha256_text


def _nullable(schema: dict[str, Any]) -> dict[str, Any]:
    return {"anyOf": [schema, {"type": "null"}]}


class AgentWorkspaceTool(Tool):
    """Private agent-owned playground, deliberately separate from user repositories."""

    name = "agent_workspace"
    description = (
        "Use the agent-owned persistent sandbox for general calculations, disposable Python experiments, "
        "temporary research artifacts, reusable agent-owned notes/knowledge, doability tests, and other "
        "private state that does not belong to one user project. Ordinary python/shell operations have no "
        "network and cannot see host/project files outside this configured agent-data root."
    )
    usage_guidance = (
        "Prefer this tool for general agent-owned scratch computation, reusable private notes/knowledge, "
        "and disposable experiments that do not belong in one user project. Keep reusable private knowledge "
        "revisable and removable rather than treating it as append-only history. Paths are relative to the "
        "private workspace. For incomplete text, continue at next_offset with expected_sha256; "
        "for directory pages continue with after_path=next_path. Shell/Python output retains exact "
        "artifacts: use read_artifact to recover beyond a preview. Output/time limits stop the command "
        "explicitly; never assume a failed or interrupted command made no changes. Use python for calculations or experiments, shell for sandboxed command-line "
        "work, and read/write/list for durable private files. Package installation is an explicit side "
        "effect and must be requested separately. Project-specific editable text, notes, configuration, "
        "source, and other trackable working material should stay in the project's existing version-control "
        "structure when that fits; inspect the repository layout and policy before creating it. Do not copy "
        "private workspace artifacts into a user repository unless they have genuinely become project material."
    )
    kind: ToolKind = "stateful"
    input_schema = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["python", "shell", "read_text", "write_text", "list_files", "pip_install"],
            },
            "code": _nullable({"type": "string"}),
            "command": _nullable({"type": "string"}),
            "path": _nullable({"type": "string"}),
            "text": _nullable({"type": "string"}),
            "packages": _nullable({"type": "array", "items": {"type": "string"}}),
            "max_chars": _nullable({"type": "integer"}),
            "start_offset": _nullable({"type": "integer"}),
            "after_path": _nullable({"type": "string"}),
            "max_files": _nullable({"type": "integer"}),
            "expected_sha256": _nullable({"type": "string"}),
        },
        "required": ["operation", "code", "command", "path", "text", "packages", "max_chars", "start_offset", "after_path", "max_files", "expected_sha256"],
        "additionalProperties": False,
    }

    def available(self, config) -> bool:
        return (
            config.agent_data.sandbox_backend == "bwrap"
            and shutil.which("bwrap") is not None
            and Path(config.agent_data.python_executable).is_absolute()
            and Path(config.agent_data.python_executable).exists()
        )

    def effective_kind(self, validated_input: dict[str, Any]) -> ToolKind:
        return "side_effect" if validated_input["operation"] == "pip_install" else "stateful"

    def execution_timeout_seconds(self, context: ToolContext) -> float | None:
        # The tool owns subprocess timeouts so it can return bounded stdout/stderr evidence.
        return None

    def validate(self, raw_input: dict[str, Any]) -> dict[str, Any]:
        operation = raw_input.get("operation")
        allowed = {"python", "shell", "read_text", "write_text", "list_files", "pip_install"}
        if operation not in allowed:
            raise ToolValidationError("agent_workspace.operation is invalid")
        code = raw_input.get("code")
        command = raw_input.get("command")
        path = raw_input.get("path")
        text = raw_input.get("text")
        packages = raw_input.get("packages")
        max_chars = raw_input.get("max_chars")
        for name, value in (("code", code), ("command", command), ("path", path), ("text", text)):
            if value is not None and not isinstance(value, str):
                raise ToolValidationError(f"agent_workspace.{name} must be a string or null")
        if packages is not None:
            if not isinstance(packages, list) or not packages:
                raise ToolValidationError("agent_workspace.packages must be a non-empty array or null")
            clean_packages: list[str] = []
            for package in packages:
                if not isinstance(package, str) or not package.strip():
                    raise ToolValidationError("agent_workspace.packages must contain non-empty strings")
                candidate = package.strip()
                if any(ch in candidate for ch in "\n\r\x00"):
                    raise ToolValidationError("agent_workspace package names contain invalid characters")
                clean_packages.append(candidate)
            packages = clean_packages
        if max_chars is not None and (
            isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars <= 0
        ):
            raise ToolValidationError("agent_workspace.max_chars must be a positive integer or null")
        start_offset = raw_input.get("start_offset") or 0
        if isinstance(raw_input.get("start_offset"), bool) or not isinstance(start_offset, int) or start_offset < 0:
            raise ToolValidationError("agent_workspace.start_offset must be a non-negative integer or null")
        max_files = raw_input.get("max_files")
        if max_files is not None and (isinstance(max_files, bool) or not isinstance(max_files, int) or max_files <= 0):
            raise ToolValidationError("agent_workspace.max_files must be a positive integer or null")
        after_path = raw_input.get("after_path")
        expected = raw_input.get("expected_sha256")
        if after_path is not None and not isinstance(after_path, str):
            raise ToolValidationError("agent_workspace.after_path must be a string or null")
        if expected is not None and (not isinstance(expected, str) or expected and (len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected))):
            raise ToolValidationError("agent_workspace.expected_sha256 must be a SHA-256, empty for absent, or null")
        if operation == "python" and not (isinstance(code, str) and code.strip()):
            raise ToolValidationError("agent_workspace.python requires non-empty code")
        if operation == "shell" and not (isinstance(command, str) and command.strip()):
            raise ToolValidationError("agent_workspace.shell requires non-empty command")
        if operation in {"read_text", "write_text", "list_files"} and path is None:
            path = "."
        if operation == "write_text" and text is None:
            raise ToolValidationError("agent_workspace.write_text requires text")
        if operation == "pip_install" and not packages:
            raise ToolValidationError("agent_workspace.pip_install requires packages")
        return {
            "operation": operation,
            "code": code or "",
            "command": command or "",
            "path": (path or ".").strip() or ".",
            "text": text or "",
            "packages": packages or [],
            "max_chars": max_chars,
            "start_offset": start_offset, "after_path": after_path or "",
            "max_files": max_files, "expected_sha256": expected,
        }

    def required_generated_event_types(self, validated_input: dict[str, Any]) -> set[str]:
        return {"agent_workspace_operation"}

    def execute(self, validated_input: dict[str, Any], context: ToolContext) -> ToolExecutionResult:
        root, workspace, home, venv = _ensure_layout(context)
        operation = validated_input["operation"]
        events: list[ToolGeneratedEvent] = []
        max_chars = min(
            validated_input["max_chars"] or context.config.agent_data.max_capture_chars,
            context.config.agent_data.max_capture_chars,
        )
        if operation == "read_text":
            target = _resolve_workspace_path(workspace, validated_input["path"])
            chars, size_bytes, digest = TextArtifactStore._measure(target, errors="strict")
            expected = validated_input["expected_sha256"]
            if expected is not None and expected != digest:
                raise ToolValidationError("Agent workspace file changed; restart reading from its current revision")
            start = min(validated_input["start_offset"], chars)
            with target.open("r", encoding="utf-8", newline="") as source:
                remaining = start
                while remaining:
                    chunk = source.read(min(remaining, 65536))
                    if not chunk:
                        raise ToolValidationError("Agent workspace file changed during reading")
                    remaining -= len(chunk)
                content = source.read(max_chars)
            if TextArtifactStore._measure(target, errors="strict")[2] != digest:
                raise ToolValidationError("Agent workspace file changed during reading")
            end = start + len(content)
            output = {
                "operation": operation, "path": _relative_name(workspace, target),
                "text": content, "truncated": end < chars, "chars": chars,
                "bytes": size_bytes, "sha256": digest, "start_offset": start,
                "next_offset": end, "finished": end >= chars,
            }
        elif operation == "write_text":
            target = _resolve_workspace_path(workspace, validated_input["path"])
            with _workspace_lock(root, "writes", context.config.agent_data.command_timeout_seconds):
                expected = validated_input["expected_sha256"]
                actual = TextArtifactStore._measure(target, errors="strict")[2] if target.exists() else ""
                if expected is not None and expected != actual:
                    raise ToolValidationError("Agent workspace file revision changed; write was not applied")
                target.parent.mkdir(parents=True, exist_ok=True)
                fd, temporary = tempfile.mkstemp(prefix=".write-", dir=target.parent)
                try:
                    with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
                        handle.write(validated_input["text"])
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(temporary, target)
                finally:
                    Path(temporary).unlink(missing_ok=True)
            output = {"operation": operation, "path": _relative_name(workspace, target),
                      "chars": len(validated_input["text"]),
                      "sha256": sha256_text(validated_input["text"])}
        elif operation == "list_files":
            target = _resolve_workspace_path(workspace, validated_input["path"])
            if not target.exists():
                raise FileNotFoundError(f"Agent workspace path not found: {validated_input['path']}")
            limit = min(validated_input["max_files"] or context.config.agent_data.max_list_files,
                        context.config.agent_data.max_list_files)
            total = 0
            def candidates():
                nonlocal total
                for item in ([target] if target.is_file() else target.rglob("*")):
                    if not item.is_file() or not item.resolve().is_relative_to(workspace):
                        continue
                    total += 1
                    name = item.relative_to(workspace).as_posix()
                    if name > validated_input["after_path"]:
                        yield name, item.stat().st_size
            selected = heapq.nsmallest(limit + 1, candidates())
            page = selected[:limit]
            output = {
                "operation": operation, "path": _relative_name(workspace, target),
                "files": [{"path": name, "bytes": size} for name, size in page],
                "total_files": total, "truncated": len(selected) > limit,
                "finished": len(selected) <= limit,
                "next_path": page[-1][0] if page else validated_input["after_path"],
                "consistency": "live directory view; concurrent changes may require restarting traversal",
            }
        elif operation == "python":
            events.extend(_ensure_venv(context, venv))
            output, capture_events = _run_sandboxed(
                context,
                root,
                ["/agent/python-venv/bin/python", "-c", validated_input["code"]],
                max_chars=max_chars,
                network=False,
            )
            output["operation"] = operation
            events.extend(capture_events)
        elif operation == "shell":
            output, capture_events = _run_sandboxed(
                context,
                root,
                ["/bin/bash", "-lc", validated_input["command"]],
                max_chars=max_chars,
                network=False,
            )
            output["operation"] = operation
            events.extend(capture_events)
        else:
            events.extend(_ensure_venv(context, venv))
            output, capture_events = _run_sandboxed(
                context,
                root,
                [
                    "/agent/python-venv/bin/python",
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    *validated_input["packages"],
                ],
                max_chars=max_chars,
                network=True,
            )
            output.update(operation=operation, packages=validated_input["packages"])
            events.extend(capture_events)
        event = ToolGeneratedEvent(
            "agent_workspace_operation",
            {
                "operation": operation,
                "path": output.get("path", ""),
                "return_code": output.get("return_code"),
                "sandboxed": True,
                "network_enabled": operation == "pip_install",
            },
        )
        events.append(event)
        if output.get("status") in {"timed_out", "output_limit_exceeded", "capture_drain_incomplete"}:
            raise ToolExecutionError(
                "Agent workspace command stopped: " + output["status"],
                error_type=output["status"], evidence=output, generated_events=events,
            )
        return ToolExecutionResult(
            tool_name=self.name,
            output=output,
            display_text=f"agent_workspace result: {stable_json_dumps(output, indent=2)}",
            generated_events=events,
        )


def _ensure_layout(context: ToolContext) -> tuple[Path, Path, Path, Path]:
    root = context.config.agent_data.root.expanduser().resolve()
    workspace = root / "workspace"
    home = root / "home"
    venv = root / "python-venv"
    for path in (root, workspace, home):
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            path.chmod(0o700)
        except OSError:
            pass
    return root, workspace, home, venv


@contextmanager
def _workspace_lock(root: Path, name: str, timeout: float, *, shared: bool = False):
    with (root / ("." + name + ".lock")).open("a") as handle:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(handle, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise ToolValidationError("Agent workspace mutation is busy; no new change was applied")
                time.sleep(0.02)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _ensure_venv(context: ToolContext, venv: Path) -> list[ToolGeneratedEvent]:
    with _workspace_lock(venv.parent, "venv", context.config.agent_data.command_timeout_seconds):
        ready = venv.parent / ".venv.ready"
        if ready.exists() and (venv / "bin/python").exists() and (venv / "pyvenv.cfg").exists():
            return []
        # A partial creation must never be treated as ready merely because its
        # interpreter symlink exists. Re-running venv completes its own layout.
        output, events = capture_command(
            [context.config.agent_data.python_executable, "-m", "venv", str(venv)],
            store=context.environment.artifacts,
            cwd=venv.parent,
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
            timeout_seconds=context.config.agent_data.command_timeout_seconds,
            max_output_bytes=context.config.agent_data.max_output_bytes,
            preview_chars=context.config.agent_data.max_capture_chars,
            kind="agent_workspace_venv",
        )
        if output["status"] != "completed" or not (venv / "bin/python").exists() or not (venv / "pyvenv.cfg").exists():
            raise ToolExecutionError("Could not create the agent Python environment", evidence=output, generated_events=events)
        with ready.open("w") as handle:
            handle.write("initialized\n"); handle.flush(); os.fsync(handle.fileno())
        return events


def _resolve_workspace_path(workspace: Path, relative: str) -> Path:
    value = Path(relative)
    if value.is_absolute():
        raise ToolValidationError("agent_workspace paths must be relative to the private workspace")
    target = (workspace / value).resolve()
    try:
        target.relative_to(workspace.resolve())
    except ValueError as exc:
        raise ToolValidationError("agent_workspace path escapes the private workspace") from exc
    return target


def _relative_name(workspace: Path, target: Path) -> str:
    try:
        value = target.resolve().relative_to(workspace.resolve())
    except ValueError:
        return "."
    text = value.as_posix()
    return "." if text == "." else text


def _bwrap_args(context: ToolContext, root: Path, *, network: bool) -> list[str]:
    args = [
        "bwrap",
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--dir", "/agent",
        "--bind", str(root / "workspace"), "/agent/workspace",
        "--bind", str(root / "home"), "/agent/home",
        "--chdir",
        "/agent/workspace",
        "--setenv",
        "HOME",
        "/agent/home",
        "--setenv",
        "PATH",
        "/agent/python-venv/bin:/usr/bin:/bin",
        "--setenv",
        "LANG",
        "C.UTF-8",
    ]
    if (root / "python-venv").exists():
        args.extend(["--bind" if network else "--ro-bind",
                     str(root / "python-venv"), "/agent/python-venv"])
    if (root / "knowledge").exists():
        args.extend(["--ro-bind", str(root / "knowledge"), "/agent/knowledge"])
    if network:
        args.append("--share-net")
    for system_path in ("/usr", "/bin", "/lib", "/lib64", "/sbin"):
        if Path(system_path).exists():
            args.extend(["--ro-bind", system_path, system_path])
    if network:
        for system_path in (
            "/etc/resolv.conf",
            "/etc/hosts",
            "/etc/nsswitch.conf",
            "/etc/ssl/certs",
        ):
            if Path(system_path).exists():
                args.extend(["--ro-bind", system_path, system_path])
    return args


def _run_sandboxed(
    context: ToolContext,
    root: Path,
    command: list[str],
    *,
    max_chars: int,
    network: bool,
) -> tuple[dict[str, Any], list[ToolGeneratedEvent]]:
    args = [*_bwrap_args(context, root, network=network), "--", *command]
    with _workspace_lock(root, "venv", context.config.agent_data.command_timeout_seconds,
                         shared=not network):
        output, events = capture_command(
            args, store=context.environment.artifacts,
            timeout_seconds=float(context.config.agent_data.command_timeout_seconds),
            max_output_bytes=context.config.agent_data.max_output_bytes,
            preview_chars=max_chars,
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
            kind="agent_workspace",
        )
    output.update(sandboxed=True, network_enabled=network)
    return output, events


AGENT_WORKSPACE_TOOLS = [AgentWorkspaceTool()]
