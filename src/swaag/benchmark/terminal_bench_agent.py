from __future__ import annotations

import json
import shlex
import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import certifi
import charset_normalizer
import idna
import requests
import urllib3
from terminal_bench.agents.base_agent import AgentResult, BaseAgent
from terminal_bench.agents.failure_mode import FailureMode

from swaag.config import load_config


class RealAgentTerminalBenchAgent(BaseAgent):
    @staticmethod
    def name() -> str:
        return "swaag-real-agent"

    @staticmethod
    def _repo_root() -> Path:
        return Path(__file__).resolve().parents[3]

    @staticmethod
    def _copy_tree(source: Path, destination: Path) -> None:
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    @staticmethod
    def _configured_model_base_url() -> str:
        configured = load_config().model.base_url.strip()
        parsed = urlsplit(configured)
        if not parsed.scheme or parsed.hostname is None:
            return configured or "http://127.0.0.1:14829"
        host = parsed.hostname
        if host in {"localhost", "0.0.0.0", "::1"}:
            host = "127.0.0.1"
        netloc = host
        if parsed.port is not None:
            netloc = f"{netloc}:{parsed.port}"
        return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))

    @classmethod
    def _container_model_base_url_template(cls) -> str:
        configured = cls._configured_model_base_url()
        parsed = urlsplit(configured)
        if not parsed.scheme or parsed.hostname is None:
            return configured or "http://__SWAAG_HOST_IP__:14829"
        host = parsed.hostname
        if host in {"127.0.0.1", "localhost", "0.0.0.0", "::1"}:
            host = "__SWAAG_HOST_IP__"
        netloc = host
        if parsed.port is not None:
            netloc = f"{netloc}:{parsed.port}"
        return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))

    def _build_bundle(self, instruction: str) -> Path:
        bundle_root = Path(tempfile.mkdtemp(prefix="swaag-terminal-bench-")) / "swaag_bundle"
        src_root = self._repo_root() / "src"
        site_root = bundle_root / "sitepkgs"
        configured_model_base_url = self._configured_model_base_url()
        model_base_url_template = self._container_model_base_url_template()
        self._copy_tree(src_root, bundle_root / "src")
        for module in (requests, urllib3, charset_normalizer, idna, certifi):
            module_path = Path(module.__file__).resolve()
            source = module_path.parent if module_path.name == "__init__.py" else module_path
            target = site_root / source.name
            self._copy_tree(source, target)
        (bundle_root / "instruction.txt").write_text(
            self._render_instruction(
                "You are running inside a local Terminal-Bench task container.\n"
                "Complete the task in the current workspace.\n"
                "Run the task tests before finishing.\n"
                "Do not ask for clarification.\n\n"
                f"Task instruction:\n{instruction.strip()}\n"
            ),
            encoding="utf-8",
        )
        (bundle_root / "run_swaag.sh").write_text(
            "\n".join(
                [
                    "#!/usr/bin/env bash",
                    "set -euo pipefail",
                    "ROOT=\"$(pwd)\"",
                    f"MODEL_BASE_URL={shlex.quote(configured_model_base_url)}",
                    f"MODEL_BASE_URL_TEMPLATE={shlex.quote(model_base_url_template)}",
                    "HOST_IP=\"$(python3 - <<'PY'\nimport socket, struct\nwith open('/proc/net/route', 'r', encoding='utf-8') as fh:\n    next(fh)\n    for line in fh:\n        fields = line.strip().split()\n        if fields[1] != '00000000' or not int(fields[3], 16) & 2:\n            continue\n        print(socket.inet_ntoa(struct.pack('<L', int(fields[2], 16))))\n        break\nPY\n)\"",
                    "export PYTHONPATH=\"/opt/src:/opt/sitepkgs:${PYTHONPATH:-}\"",
                    "if [[ \"${SWAAG_TERMINAL_HOST_NETWORK:-0}\" == \"1\" ]]; then",
                    "  export SWAAG__MODEL__BASE_URL=\"$MODEL_BASE_URL\"",
                    "else",
                    "  export SWAAG__MODEL__BASE_URL=\"${MODEL_BASE_URL_TEMPLATE/__SWAAG_HOST_IP__/$HOST_IP}\"",
                    "fi",
                    "export SWAAG__SESSIONS__ROOT=\"/tmp/swaag_sessions\"",
                    "export SWAAG__TOOLS__READ_ROOTS=\"[\\\"${ROOT}\\\"]\"",
                    "export SWAAG__TOOLS__ALLOW_SIDE_EFFECT_TOOLS=true",
                    "export SWAAG__TOOLS__ALLOW_STATEFUL_TOOLS=true",
                    "python3 -m swaag ask --session terminal-bench-real-agent < /opt/instruction.txt",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        (bundle_root / "run_swaag.sh").chmod(0o755)
        return bundle_root

    def perform_task(self, instruction: str, session, logging_dir: Path | None = None) -> AgentResult:
        bundle_root = self._build_bundle(instruction)
        try:
            session.copy_to_container(bundle_root, container_dir="/opt")
            session.send_keys(
                ["bash /opt/run_swaag.sh", "Enter"],
                block=True,
                max_timeout_sec=float("inf"),
            )
            return AgentResult(
                total_input_tokens=0,
                total_output_tokens=0,
                failure_mode=FailureMode.NONE,
            )
        except TimeoutError:
            return AgentResult(
                total_input_tokens=0,
                total_output_tokens=0,
                failure_mode=FailureMode.AGENT_TIMEOUT,
            )
        except Exception:
            return AgentResult(
                total_input_tokens=0,
                total_output_tokens=0,
                failure_mode=FailureMode.UNKNOWN_AGENT_ERROR,
            )
        finally:
            shutil.rmtree(bundle_root.parent, ignore_errors=True)
