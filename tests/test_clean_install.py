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

        if "grammar" in body:
            self._json_response({"content": "yes", "stop": True, "tokens_evaluated": 2, "tokens_predicted": 1})
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
        if {"task_type", "completeness", "requires_expansion", "requires_decomposition", "confidence", "detected_entities", "detected_goals"} <= properties:
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
        elif properties == {"scores"}:
            count = int(schema["properties"]["scores"].get("minItems", 0))
            payload = {"scores": [1.0] * count}
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
                "tool_input": {"expression": "6 * 7"},
            }
        elif properties == {"criteria"}:
            names = schema["properties"]["criteria"]["items"]["properties"]["name"]["enum"]
            payload = {
                "criteria": [{"name": name, "passed": True, "evidence": "criterion met"} for name in names]
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


def _create_clean_room(tmp_path: Path) -> tuple[Path, Path, dict[str, str], HTTPServer, threading.Thread]:
    repo_root = Path(__file__).resolve().parents[1]
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    venv_dir = tmp_path / "venv"

    builder = venv.EnvBuilder(with_pip=True, system_site_packages=True)
    builder.create(venv_dir)
    python = _venv_python(venv_dir)

    subprocess.run(
        [str(python), "-m", "pip", "install", "-q", "--no-build-isolation", str(repo_root)],
        check=True,
        cwd=workspace,
    )

    server = HTTPServer(("127.0.0.1", 0), _CleanRoomHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    env = os.environ.copy()
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
        assert doctor_payload["grammar_probe"] == "yes"

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
        assert total_count >= 170
        assert generated_count >= 150

        single_task = subprocess.run(
            [
                str(python),
                "-m",
                "swaag.benchmark",
                "run",
                "--clean",
                "--task",
                "coding_generated_multifile_01",
                "--output",
                str(tmp_path / "single_benchmark_output"),
                "--json",
            ],
            check=True,
            cwd=workspace,
            env=env,
            text=True,
            capture_output=True,
        )
        single_payload = json.loads(single_task.stdout)
        assert single_payload["summary"]["failed_tasks"] == 0
        assert single_payload["summary"]["successful_tasks"] == 1

        devcheck = subprocess.run(
            [str(python), "-m", "swaag.devcheck", "--dry-run", "--changed-file", "src/swaag/runtime.py"],
            check=True,
            cwd=workspace,
            env=env,
            text=True,
            capture_output=True,
        )
        assert "tests/test_runtime_verification_flow.py" in devcheck.stdout
        assert "test_profile=system" in devcheck.stdout
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
        assert "--validation-subset" in finalproof.stdout
        assert "--model-profile small_fast" in finalproof.stdout
        assert "--structured-output-mode post_validate" in finalproof.stdout
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


@pytest.mark.agent_test
def test_package_runs_full_benchmark_from_clean_venv(tmp_path: Path) -> None:
    python, workspace, env, server, thread = _create_clean_room(tmp_path)
    try:
        benchmark = subprocess.run(
            [
                str(python),
                "-m",
                "swaag.benchmark",
                "run",
                "--clean",
                "--output",
                str(tmp_path / "benchmark_output"),
                "--json",
            ],
            check=True,
            cwd=workspace,
            env=env,
            text=True,
            capture_output=True,
        )
        benchmark_payload = json.loads(benchmark.stdout)
        assert benchmark_payload["summary"]["failed_tasks"] == 0
        assert benchmark_payload["summary"]["false_positives"] == 0
        assert (tmp_path / "benchmark_output" / "benchmark_results.json").exists()
        assert (tmp_path / "benchmark_output" / "benchmark_report.md").exists()
    finally:
        server.shutdown()
        thread.join(timeout=5)
