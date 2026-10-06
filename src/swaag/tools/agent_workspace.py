from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from swaag.tools.base import Tool, ToolContext, ToolValidationError
from swaag.types import ToolExecutionResult, ToolGeneratedEvent, ToolKind
from swaag.utils import stable_json_dumps


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
        "private workspace. Use python for calculations or experiments, shell for sandboxed command-line "
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
        },
        "required": ["operation", "code", "command", "path", "text", "packages", "max_chars"],
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
        }

    def required_generated_event_types(self, validated_input: dict[str, Any]) -> set[str]:
        return {"agent_workspace_operation"}

    def execute(self, validated_input: dict[str, Any], context: ToolContext) -> ToolExecutionResult:
        root, workspace, home, venv = _ensure_layout(context)
        operation = validated_input["operation"]
        max_chars = min(
            validated_input["max_chars"] or context.config.agent_data.max_capture_chars,
            context.config.agent_data.max_capture_chars,
        )
        if operation == "read_text":
            target = _resolve_workspace_path(workspace, validated_input["path"])
            content = target.read_text(encoding="utf-8")
            output = {
                "operation": operation,
                "path": _relative_name(workspace, target),
                "text": content[:max_chars],
                "truncated": len(content) > max_chars,
                "chars": len(content),
            }
        elif operation == "write_text":
            target = _resolve_workspace_path(workspace, validated_input["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(validated_input["text"], encoding="utf-8")
            output = {
                "operation": operation,
                "path": _relative_name(workspace, target),
                "chars": len(validated_input["text"]),
            }
        elif operation == "list_files":
            target = _resolve_workspace_path(workspace, validated_input["path"])
            if not target.exists():
                raise FileNotFoundError(f"Agent workspace path not found: {validated_input['path']}")
            if target.is_file():
                items = [target]
            else:
                items = sorted(item for item in target.rglob("*") if item.is_file())
            output = {
                "operation": operation,
                "path": _relative_name(workspace, target),
                "files": [
                    {"path": _relative_name(workspace, item), "bytes": item.stat().st_size}
                    for item in items[:1000]
                ],
                "total_files": len(items),
                "truncated": len(items) > 1000,
            }
        elif operation == "python":
            _ensure_venv(context, venv)
            output = _run_sandboxed(
                context,
                root,
                ["/agent/python-venv/bin/python", "-c", validated_input["code"]],
                max_chars=max_chars,
                network=False,
            ) | {"operation": operation}
        elif operation == "shell":
            output = _run_sandboxed(
                context,
                root,
                ["/bin/bash", "-lc", validated_input["command"]],
                max_chars=max_chars,
                network=False,
            ) | {"operation": operation}
        else:
            _ensure_venv(context, venv)
            output = _run_sandboxed(
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
            ) | {"operation": operation, "packages": validated_input["packages"]}
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
        return ToolExecutionResult(
            tool_name=self.name,
            output=output,
            display_text=f"agent_workspace result: {stable_json_dumps(output, indent=2)}",
            generated_events=[event],
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


def _ensure_venv(context: ToolContext, venv: Path) -> None:
    executable = venv / "bin/python"
    if executable.exists():
        return
    result = subprocess.run(
        [context.config.agent_data.python_executable, "-m", "venv", str(venv)],
        cwd=str(context.config.agent_data.root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=float(context.config.agent_data.command_timeout_seconds),
        check=False,
    )
    if result.returncode != 0 or not executable.exists():
        raise RuntimeError(
            "Could not create the private agent Python environment: "
            + (result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}")
        )


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
        "--bind",
        str(root),
        "/agent",
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
) -> dict[str, Any]:
    args = [*_bwrap_args(context, root, network=network), "--", *command]
    completed = subprocess.run(
        args,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=float(context.config.agent_data.command_timeout_seconds),
        check=False,
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
    )
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    return {
        "return_code": completed.returncode,
        "stdout": stdout[:max_chars],
        "stderr": stderr[:max_chars],
        "stdout_truncated": len(stdout) > max_chars,
        "stderr_truncated": len(stderr) > max_chars,
        "sandboxed": True,
        "network_enabled": network,
    }


AGENT_WORKSPACE_TOOLS = [AgentWorkspaceTool()]
