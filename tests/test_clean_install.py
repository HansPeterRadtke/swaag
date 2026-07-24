from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import venv
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from tests.helpers import plan_response, plan_step


class _CleanRoomHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args):  # noqa: A003
        return

    def _json_response(self, payload: dict, status: int = 200) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._json_response({"status": "ok"})
            return
        self._json_response({"error": "not found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        if self.path == "/tokenize":
            content = body.get("content", "")
            self._json_response({"tokens": list(range(len(str(content).split())))})
            return
        if self.path != "/completion":
            self._json_response({"error": "not found"}, status=404)
            return

        schema = body.get("json_schema") or {}
        if not schema:
            prompt = str(body.get("prompt", ""))
            if "Return JSON only" in prompt and "action=" in prompt and "tool_name=" in prompt:
                self._json_response(
                    {
                        "content": json.dumps(
                            {
                                "action": "respond",
                                "response": "ok",
                                "tool_name": "none",
                                "tool_input": {},
                            }
                        ),
                        "stop": True,
                        "tokens_evaluated": 8,
                        "tokens_predicted": 8,
                    }
                )
                return
            self._json_response(
                {
                    "content": "42",
                    "stop": True,
                    "tokens_evaluated": 8,
                    "tokens_predicted": 1,
                }
            )
            return
        properties = set((schema.get("properties") or {}).keys())
        payload: dict
        if properties == {"answer"}:
            payload = {"answer": "yes"}
        elif properties == {"text"}:
            payload = {"text": "42"}
        elif properties == {"expression"}:
            payload = {"expression": "6 * 7"}
        elif {"task_type", "completeness", "requires_expansion", "requires_decomposition", "confidence", "detected_entities", "detected_goals"} <= properties:
            payload = {
                "task_type": "structured",
                "completeness": "complete",
                "requires_expansion": False,
                "requires_decomposition": True,
                "confidence": 0.99,
                "detected_entities": [],
                "detected_goals": ["compute"],
            }
        elif {"split_task", "expand_task", "ask_user", "assume_missing", "generate_ideas", "confidence", "reason"} <= properties:
            payload = {
                "split_task": True,
                "expand_task": False,
                "ask_user": False,
                "assume_missing": False,
                "generate_ideas": False,
                "direct_response": False,
                "execution_mode": "full_plan",
                "preferred_tool_name": "",
                "confidence": 0.99,
                "reason": "structured calculator task",
            }
        elif {"goal", "success_criteria", "fallback_strategy", "steps"} <= properties:
            payload = json.loads(
                plan_response(
                    goal="Use calculator to compute 6 * 7",
                    steps=[
                        plan_step(
                            "step_calc",
                            "Compute 6 * 7",
                            "tool",
                            expected_tool="calculator",
                            expected_output="42",
                            success_criteria="Calculator returns 42",
                            verification_checks=[
                                {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                                {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
                                {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "calculator"},
                                {"name": "exact_result", "check_type": "exact_match", "actual_source": "tool_output.result", "expected": 42},
                            ],
                            required_conditions=["dependencies_completed", "tool_result_present", "tool_name_matches", "exact_result"],
                            optional_conditions=[],
                        ),
                        plan_step(
                            "step_answer",
                            "Return the answer",
                            "respond",
                            expected_output="42",
                            success_criteria="Answer is 42",
                            depends_on=["step_calc"],
                        ),
                    ],
                )
            )
        elif {"task_profile", "strategy_name", "explore_before_commit", "tool_chain_depth", "verification_intensity", "reason"} <= properties:
            payload = {
                "task_profile": "multi_step",
                "strategy_name": "conservative",
                "explore_before_commit": False,
                "tool_chain_depth": 1,
                "verification_intensity": 0.9,
                "reason": "calculator workflow is short and deterministic",
            }
        elif {"spawn", "subagent_type", "reason", "focus"} <= properties:
            payload = {
                "spawn": False,
                "subagent_type": "none",
                "reason": "no specialist required for clean-room calculator task",
                "focus": "",
            }
        elif {"output_class", "reason", "units"} <= properties:
            payload = {
                "output_class": "open_ended",
                "reason": "one final answer unit is sufficient",
                "units": [
                    {
                        "unit_id": "answer_unit_01",
                        "title": "Final answer",
                        "instruction": "Provide the final user-facing answer.",
                    }
                ],
            }
        elif {"keep_partial", "reason", "next_units"} <= properties:
            payload = {
                "keep_partial": True,
                "reason": "keep the current partial answer",
                "next_units": [],
            }
        elif properties and all(key.startswith("score_") for key in properties):
            payload = {key: 1.0 for key in properties}
        elif properties == {"scores"}:  # legacy transport compatibility
            prompt = str(body.get("prompt", ""))
            if not prompt and isinstance(body.get("messages"), list):
                prompt = "\n".join(str(item.get("content", "")) for item in body["messages"] if isinstance(item, dict))
            count = sum(1 for line in prompt.splitlines() if line.startswith("[") and "]" in line)
            payload = {"scores": [1.0] * count}
        elif {
            "decision_matches_request",
            "decision_is_internally_consistent",
            "required_evidence_sources",
            "minimum_evidence_call_count",
            "selected_mode_and_tool_can_cover_declared_count",
            "feedback",
        } <= properties:
            payload = {
                "decision_matches_request": True,
                "decision_is_internally_consistent": True,
                "required_evidence_sources": [],
                "minimum_evidence_call_count": 0,
                "selected_mode_and_tool_can_cover_declared_count": True,
                "feedback": "clean-room decision is internally consistent",
            }
        elif {"kind", "retryable", "requires_replan", "suggested_strategy_mode", "wait_seconds", "reason"} <= properties:
            payload = {
                "kind": "deterministic_permanent",
                "retryable": False,
                "requires_replan": True,
                "suggested_strategy_mode": "conservative",
                "wait_seconds": 0,
                "reason": "clean-room fake server classified a deterministic failure",
            }
        elif {"action", "reason"} <= properties:
            payload = {"action": "execute_step", "reason": "single ready step"}
        elif {"action", "response", "tool_name", "tool_input"} <= properties:
            payload = {
                "action": "call_tool",
                "response": "",
                "tool_name": "calculator",
                "tool_input": {},
            }
        elif properties == {"criteria"}:
            item_properties = schema["properties"]["criteria"]["items"]["properties"]
            names = item_properties["name"]["enum"]
            id_schema = item_properties.get("candidate_excerpt_id_1", {})
            allowed_ids = id_schema.get("enum", []) if isinstance(id_schema, dict) else []
            excerpt_id = next((item for item in allowed_ids if item), "")
            payload = {
                "criteria": [
                    {
                        "name": name,
                        "passed": True,
                        "evidence": "criterion met",
                        "candidate_excerpt_id_1": excerpt_id,
                        "candidate_excerpt_id_2": "",
                        "candidate_excerpt_id_3": "",
                    }
                    for name in names
                ]
            }
        elif properties == {"summary"}:
            payload = {"summary": "summary"}
        elif {"original_goal", "expanded_goal", "scope", "constraints", "expected_outputs", "assumptions"} <= properties:
            payload = {
                "original_goal": "Use calculator to compute 6 * 7",
                "expanded_goal": "Use calculator to compute 6 * 7",
                "scope": ["single calculation"],
                "constraints": ["return only the numeric result"],
                "expected_outputs": ["42"],
                "assumptions": [],
            }
        else:
            raise AssertionError(f"Unhandled schema properties: {sorted(properties)}")

        self._json_response(
            {
                "content": json.dumps(payload),
                "stop": True,
                "tokens_evaluated": 12,
                "tokens_predicted": 24,
            }
        )


def _venv_python(venv_dir: Path) -> Path:
    return venv_dir / "bin" / "python"


def _clean_python_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    return env


def _create_clean_room(tmp_path: Path) -> tuple[Path, Path, dict[str, str], HTTPServer, threading.Thread]:
    repo_root = Path(__file__).resolve().parents[1]
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    venv_dir = tmp_path / "venv"

    builder = venv.EnvBuilder(with_pip=True, system_site_packages=True)
    base_executable = str(Path(getattr(sys, "_base_executable", sys.executable)).resolve())
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.delenv("PYTHONHOME", raising=False)
        monkeypatch.delenv("PYTHONPATH", raising=False)
        monkeypatch.setattr(sys, "_base_executable", base_executable, raising=False)
        builder.create(venv_dir)
    python = _venv_python(venv_dir)

    subprocess.run(
        [str(python), "-m", "pip", "install", "-q", "setuptools>=68", "wheel"],
        check=True,
        cwd=workspace,
        env=_clean_python_subprocess_env(),
    )
    subprocess.run(
        [str(python), "-m", "pip", "install", "-q", "--no-build-isolation", str(repo_root)],
        check=True,
        cwd=workspace,
        env=_clean_python_subprocess_env(),
    )

    server = HTTPServer(("127.0.0.1", 0), _CleanRoomHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    env = _clean_python_subprocess_env()
    env.update(
        {
            "SWAAG__MODEL__BASE_URL": f"http://127.0.0.1:{server.server_port}",
            "SWAAG__SESSIONS__ROOT": str(tmp_path / "sessions"),
            "SWAAG__TOOLS__READ_ROOTS": json.dumps([str(tmp_path)]),
        }
    )
    return python, workspace, env, server, thread


def test_package_installs_and_cli_runs_from_clean_venv(tmp_path: Path) -> None:
    python, workspace, env, server, thread = _create_clean_room(tmp_path)
    try:
        doctor = subprocess.run(
            [str(python), "-m", "swaag", "doctor", "--json"],
            check=True,
            cwd=workspace,
            env=env,
            text=True,
            capture_output=True,
        )
        doctor_payload = json.loads(doctor.stdout)
        assert doctor_payload["health"]["status"] == "ok"
        assert doctor_payload["json_probe"] == "yes"

        ask = subprocess.run(
            [str(python), "-m", "swaag", "ask", "Use calculator to compute 6 * 7"],
            check=True,
            cwd=workspace,
            env=env,
            text=True,
            capture_output=True,
        )
        assert ask.stdout.strip() == "42"
        assert "[session=" in ask.stderr

        catalog = subprocess.run(
            [
                str(python),
                "-c",
                (
                    "from swaag.benchmark.scaled_catalog import generated_benchmark_tasks; "
                    "from swaag.benchmark.task_definitions import get_benchmark_tasks, validate_benchmark_catalog; "
                    "tasks = get_benchmark_tasks(); "
                    "validate_benchmark_catalog(tasks); "
                    "print(len(tasks), len(generated_benchmark_tasks()))"
                ),
            ],
            check=True,
            cwd=workspace,
            env=env,
            text=True,
            capture_output=True,
        )
        total_count, generated_count = [int(part) for part in catalog.stdout.strip().split()]
        assert total_count >= 50
        assert generated_count >= 24

        devcheck = subprocess.run(
            [str(python), "-m", "swaag.devcheck", "--dry-run", "--changed-file", "src/swaag/runtime.py"],
            check=True,
            cwd=workspace,
            env=env,
            text=True,
            capture_output=True,
        )
        assert "tests/test_runtime_verification_flow.py" in devcheck.stdout
        assert "candidate_tests=" in devcheck.stdout
        assert "testmon=available:False" in devcheck.stdout

        finalproof = subprocess.run(
            [str(python), "-m", "swaag.finalproof", "--dry-run"],
            check=True,
            cwd=workspace,
            env=env,
            text=True,
            capture_output=True,
        )
        assert "tests/test_scaled_catalog.py" in finalproof.stdout
        assert "swaag.manual_validation" in finalproof.stdout
        assert "--full-catalog" in finalproof.stdout
        assert "--model-profile small_fast" in finalproof.stdout
        assert "--structured-output-mode server_schema" in finalproof.stdout
        assert "--seeds 11,23,37" in finalproof.stdout

        live_subset_catalog = subprocess.run(
            [
                str(python),
                "-c",
                (
                    "from swaag.benchmark.scaled_catalog import generated_live_subset_tasks; "
                    "tasks = generated_live_subset_tasks(); "
                    "print(len(tasks), len({task.task_id for task in tasks}))"
                ),
            ],
            check=True,
            cwd=workspace,
            env=env,
            text=True,
            capture_output=True,
        )
        live_total, live_unique = [int(part) for part in live_subset_catalog.stdout.strip().split()]
        assert live_total >= 30
        assert live_total == live_unique
    finally:
        server.shutdown()
        thread.join(timeout=5)
