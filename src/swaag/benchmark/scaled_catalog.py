from __future__ import annotations

import json
from pathlib import Path

from swaag.benchmark.task_definitions import (
    BenchmarkTaskDefinition,
    BenchmarkVerificationContract,
    PromptUnderstandingOracle,
    ScriptedBenchmarkClient,
    TaskScenario,
    _contract_scripted_client,
    _plan_response,
    _plan_step,
    _tool_call,
    _write,
)
from swaag.utils import stable_json_dumps


def _class_name(name: str) -> str:
    return "".join(part.capitalize() for part in name.replace("-", "_").split("_")) or "Generated"


def _coding_multifile_definition(index: int, *, environment: bool = False) -> BenchmarkTaskDefinition:
    task_id = f"coding_generated_multifile_{index:02d}"
    topic = f"package_{index:02d}"
    label = f"result{index:02d}"
    base_value = 20 + index
    delta = 3 + (index % 5)
    expected_total = base_value + delta
    wrong_base = base_value - 2
    wrong_delta = delta + 1
    answer = f"fixed-{index:02d}"
    use_write_file = environment
    use_read_file = environment
    use_run_tests = environment

    def _build(workspace: Path) -> TaskScenario:
        package_dir = workspace / topic
        _write(package_dir / "__init__.py", "")
        core = _write(package_dir / "core.py", f"def base_value() -> int:\n    return {wrong_base}\n")
        calc = _write(
            package_dir / "calc.py",
            f"from {topic}.core import base_value\n\n\ndef total() -> int:\n    return base_value() + {wrong_delta}\n",
        )
        report = _write(
            package_dir / "report.py",
            f"from {topic}.calc import total\n\n\ndef describe() -> str:\n    return f\"{label}={{total() - 1}}\"\n",
        )
        fixed_core = f"def base_value() -> int:\n    return {base_value}\n"
        fixed_calc = f"from {topic}.core import base_value\n\n\ndef total() -> int:\n    return base_value() + {delta}\n"
        fixed_report = f"from {topic}.calc import total\n\n\ndef describe() -> str:\n    return f\"{label}={{total()}}\"\n"
        test_file = f"test_{topic}.py"
        _write(
            workspace / test_file,
            (
                "import unittest\n\n"
                f"from {topic}.core import base_value\n"
                f"from {topic}.calc import total\n"
                f"from {topic}.report import describe\n\n\n"
                f"class {_class_name(topic)}Tests(unittest.TestCase):\n"
                f"    def test_base_value(self) -> None:\n        self.assertEqual(base_value(), {base_value})\n\n"
                f"    def test_total(self) -> None:\n        self.assertEqual(total(), {expected_total})\n\n"
                f"    def test_report(self) -> None:\n        self.assertEqual(describe(), \"{label}={expected_total}\")\n\n\n"
                "if __name__ == '__main__':\n    unittest.main()\n"
            ),
        )
        prompt = (
            f"Read {core}, {calc}, and {report}. Fix the multi-file project so the tests pass, "
            f"keep the modules consistent, and reply {answer}."
        )
        read_tool = "read_file" if use_read_file else "read_text"
        write_tool = "write_file" if use_write_file else "edit_text"
        steps = [
            _plan_step("step_read_core", "Read core module", "read", expected_tool=read_tool, expected_output="core.py contents", success_criteria="core.py is available."),
            _plan_step("step_read_calc", "Read calc module", "read", expected_tool=read_tool, expected_output="calc.py contents", success_criteria="calc.py is available.", depends_on=["step_read_core"]),
            _plan_step("step_read_report", "Read report module", "read", expected_tool=read_tool, expected_output="report.py contents", success_criteria="report.py is available.", depends_on=["step_read_calc"]),
            _plan_step(
                "step_fix_core",
                "Fix base_value",
                "write",
                expected_tool=write_tool,
                expected_output="Updated core.py",
                success_criteria="core.py returns the correct base value.",
                depends_on=["step_read_report"],
                verification_checks=[
                    {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                    {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
                    {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": write_tool},
                    {"name": "file_contains_expected_base", "check_type": "file_contains", "path": core, "pattern": f"return {base_value}"},
                ],
                required_conditions=["dependencies_completed", "tool_result_present", "tool_name_matches", "file_contains_expected_base"],
                optional_conditions=[],
            ),
            _plan_step(
                "step_fix_calc",
                "Fix total calculation",
                "write",
                expected_tool=write_tool,
                expected_output="Updated calc.py",
                success_criteria="calc.py computes the correct total.",
                depends_on=["step_fix_core"],
                verification_checks=[
                    {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                    {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
                    {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": write_tool},
                    {"name": "file_contains_expected_delta", "check_type": "file_contains", "path": calc, "pattern": f"return base_value() + {delta}"},
                ],
                required_conditions=["dependencies_completed", "tool_result_present", "tool_name_matches", "file_contains_expected_delta"],
                optional_conditions=[],
            ),
            _plan_step(
                "step_fix_report",
                "Fix report formatting",
                "write",
                expected_tool=write_tool,
                expected_output="Updated report.py",
                success_criteria="report.py renders the correct formatted total.",
                depends_on=["step_fix_calc"],
                verification_checks=[
                    {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                    {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
                    {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": write_tool},
                    {"name": "file_contains_expected_report", "check_type": "file_contains", "path": report, "pattern": f'return f\"{label}={{total()}}\"'},
                ],
                required_conditions=["dependencies_completed", "tool_result_present", "tool_name_matches", "file_contains_expected_report"],
                optional_conditions=[],
            ),
        ]
        responses = [
            _plan_response(goal=prompt, steps=steps + ([
                _plan_step(
                    "step_run_tests",
                    "Run unittest suite",
                    "tool",
                    expected_tool="run_tests",
                    expected_output="Passing unittest result",
                    success_criteria="The unittest suite passes.",
                    depends_on=["step_fix_report"],
                    verification_checks=[
                        {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                        {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
                        {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "run_tests"},
                        {"name": "tests_pass", "check_type": "exact_match", "actual_source": "tool_output.passed", "expected": True},
                    ],
                    required_conditions=["dependencies_completed", "tool_result_present", "tool_name_matches", "tests_pass"],
                    optional_conditions=[],
                )
            ] if use_run_tests else []) + [
                _plan_step(
                    "step_answer",
                    "Answer the user",
                    "respond",
                    expected_output=answer,
                    success_criteria=f"The assistant replies {answer}.",
                    depends_on=["step_run_tests" if use_run_tests else "step_fix_report"],
                )
            ]),
            _tool_call(read_tool, {"path": core}),
            _tool_call(read_tool, {"path": calc}),
            _tool_call(read_tool, {"path": report}),
        ]
        if use_write_file:
            responses.extend(
                [
                    _tool_call("write_file", {"path": core, "content": fixed_core}),
                    _tool_call("write_file", {"path": calc, "content": fixed_calc}),
                    _tool_call("write_file", {"path": report, "content": fixed_report}),
                    _tool_call("run_tests", {"command": ["python3", "-m", "unittest", "-q", test_file]}),
                ]
            )
        else:
            responses.extend(
                [
                    _tool_call("edit_text", {"path": core, "operation": "replace_pattern_once", "pattern": f"return {wrong_base}", "replacement": f"return {base_value}"}),
                    _tool_call("edit_text", {"path": calc, "operation": "replace_pattern_once", "pattern": f"return base_value() + {wrong_delta}", "replacement": f"return base_value() + {delta}"}),
                    _tool_call("edit_text", {"path": report, "operation": "replace_pattern_once", "pattern": f'return f\"{label}={{total() - 1}}\"', "replacement": f'return f\"{label}={{total()}}\"'}),
                ]
            )
        responses.append(answer)
        return TaskScenario(
            prompt=prompt,
            workspace=workspace,
            model_client=ScriptedBenchmarkClient(responses=responses),
            verification_contract=BenchmarkVerificationContract(
                task_type="coding",
                expected_answer=answer,
                expected_files={core: fixed_core, calc: fixed_calc, report: fixed_report},
                command=["python3", "-m", "unittest", "-q", test_file],
                command_cwd=str(workspace),
                command_framework="unittest",
                required_history_events=["tool_called", "tool_result", "verification_passed"],
                required_tools_used=[read_tool, write_tool],
                forbidden_tools_used=[] if index % 6 == 0 else ["echo"],
                min_tool_calls=7 if use_run_tests else 6,
                forbid_unexpected_workspace_changes=True,
            ),
        )

    return BenchmarkTaskDefinition(
        task_id=task_id,
        task_type="coding",
        description=f"Fix a realistic three-module code package and keep execution tests green ({'environment' if environment else 'editor'} mode).",
        build=_build,
        difficulty="hard" if environment or index % 5 == 0 else "medium",
        tags=["coding", "multifile", "realistic-code", "project-consistency", *( ["environment", "run-tests"] if environment else ["editor"] )],
        setup_instructions=[
            "Create a package with core, calc, and report modules.",
            "Create a unittest file that verifies the modules remain consistent.",
            "Require coordinated fixes across multiple source files.",
        ],
        config_overrides={
            "tools_allow_side_effect_tools": True,
            "planner_max_plan_steps": 10,
            **({"runtime_max_tool_steps": 8, "runtime_tool_call_budget": 8} if environment else {}),
        },
    )


def _file_edit_definition(index: int, *, mode: str) -> BenchmarkTaskDefinition:
    task_id = f"file_edit_generated_{mode}_{index:02d}"
    answer = {
        "exact": f"file-updated-{index:02d}",
        "replace_all": f"all-updated-{index:02d}",
        "reread": f"confirmed-{index:02d}",
    }[mode]

    def _build(workspace: Path) -> TaskScenario:
        path = workspace / f"document_{index:02d}.txt"
        if mode == "exact":
            original = f"title=report-{index:02d}\nstatus=draft\n"
            fixed = f"title=report-{index:02d}\nstatus=ready\n"
            _write(path, original)
            prompt = f"Read {path}, replace draft with ready, and reply {answer}."
            plan = _plan_response(
                goal=prompt,
                steps=[
                    _plan_step("step_read", "Read the document", "read", expected_tool="read_text", expected_output="Document text", success_criteria="The document is read."),
                    _plan_step(
                        "step_edit",
                        "Update the status",
                        "write",
                        expected_tool="edit_text",
                        expected_output="Updated document",
                        success_criteria="The document contains ready instead of draft.",
                        depends_on=["step_read"],
                        verification_checks=[
                            {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                            {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
                            {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"},
                            {"name": "file_contains_ready", "check_type": "file_contains", "path": str(path), "pattern": "status=ready"},
                        ],
                        required_conditions=["dependencies_completed", "tool_result_present", "tool_name_matches", "file_contains_ready"],
                        optional_conditions=[],
                    ),
                    _plan_step("step_answer", "Answer the user", "respond", expected_output=answer, success_criteria=f"The assistant replies {answer}.", depends_on=["step_edit"]),
                ],
            )
            responses = [
                plan,
                _tool_call("read_text", {"path": str(path)}),
                _tool_call("edit_text", {"path": str(path), "operation": "replace_pattern_once", "pattern": "status=draft", "replacement": "status=ready"}),
                answer,
            ]
            contract = BenchmarkVerificationContract(task_type="file_edit", expected_answer=answer, expected_files={str(path): fixed}, required_history_events=["edit_applied", "verification_passed"], forbid_unexpected_workspace_changes=True)
        elif mode == "replace_all":
            original = "\n".join([f"item=old-{index:02d}" for _ in range(3)]) + "\n"
            fixed = original.replace(f"old-{index:02d}", f"new-{index:02d}")
            _write(path, original)
            prompt = f"Read {path}, replace every old-{index:02d} with new-{index:02d}, and reply {answer}."
            plan = _plan_response(
                goal=prompt,
                steps=[
                    _plan_step("step_read", "Read the document", "read", expected_tool="read_text", expected_output="Document text", success_criteria="The document is read."),
                    _plan_step(
                        "step_edit",
                        "Replace every occurrence",
                        "write",
                        expected_tool="edit_text",
                        expected_output="Updated document",
                        success_criteria="Every occurrence is replaced.",
                        depends_on=["step_read"],
                        verification_checks=[
                            {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                            {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
                            {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"},
                            {"name": "file_contains_new", "check_type": "file_contains", "path": str(path), "pattern": f"new-{index:02d}"},
                        ],
                        required_conditions=["dependencies_completed", "tool_result_present", "tool_name_matches", "file_contains_new"],
                        optional_conditions=[],
                    ),
                    _plan_step("step_answer", "Answer the user", "respond", expected_output=answer, success_criteria=f"The assistant replies {answer}.", depends_on=["step_edit"]),
                ],
            )
            responses = [
                plan,
                _tool_call("read_text", {"path": str(path)}),
                _tool_call("edit_text", {"path": str(path), "operation": "replace_pattern_all", "pattern": f"old-{index:02d}", "replacement": f"new-{index:02d}"}),
                answer,
            ]
            contract = BenchmarkVerificationContract(task_type="file_edit", expected_answer=answer, expected_files={str(path): fixed}, required_history_events=["edit_applied", "verification_passed"], forbid_unexpected_workspace_changes=True)
        elif mode == "reread":
            source = workspace / f"source_{index:02d}.txt"
            target = workspace / f"target_{index:02d}.txt"
            _write(source, f"release={index:02d}.0\n")
            _write(target, "release=pending\n")
            prompt = f"Read {source}, update {target} to the same content, reread it, and reply {answer}."
            plan = _plan_response(
                goal=prompt,
                steps=[
                    _plan_step("step_read_source", "Read the source file", "read", expected_tool="read_file", expected_output="Source text", success_criteria="The source file is read."),
                    _plan_step(
                        "step_write_target",
                        "Write the target file",
                        "write",
                        expected_tool="write_file",
                        expected_output="Updated target file",
                        success_criteria="The target file matches the source file.",
                        depends_on=["step_read_source"],
                        verification_checks=[
                            {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                            {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
                            {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "write_file"},
                            {"name": "file_contains_release", "check_type": "file_contains", "path": str(target), "pattern": f"release={index:02d}.0"},
                        ],
                        required_conditions=["dependencies_completed", "tool_result_present", "tool_name_matches", "file_contains_release"],
                        optional_conditions=[],
                    ),
                    _plan_step("step_confirm_target", "Reread the target file", "read", expected_tool="read_file", expected_output="Target text", success_criteria="The target file is reread.", depends_on=["step_write_target"]),
                    _plan_step("step_answer", "Answer the user", "respond", expected_output=answer, success_criteria=f"The assistant replies {answer}.", depends_on=["step_confirm_target"]),
                ],
            )
            responses = [
                plan,
                _tool_call("read_file", {"path": str(source)}),
                _tool_call("write_file", {"path": str(target), "content": f"release={index:02d}.0\n"}),
                _tool_call("read_file", {"path": str(target)}),
                answer,
            ]
            contract = BenchmarkVerificationContract(task_type="file_edit", expected_answer=answer, expected_files={str(target): f"release={index:02d}.0\n"}, required_tools_used=["read_file", "write_file"], min_tool_calls=3, required_history_events=["filesystem_read", "file_write_applied", "verification_passed"], forbid_unexpected_workspace_changes=True)
        return TaskScenario(prompt=prompt, workspace=workspace, model_client=ScriptedBenchmarkClient(responses=responses), verification_contract=contract)

    tags = ["file-edit", mode]
    if mode == "reread":
        tags.append("quality")
    return BenchmarkTaskDefinition(
        task_id=task_id,
        task_type="file_edit",
        description=f"Generated file-edit benchmark in {mode} mode.",
        build=_build,
        difficulty="medium" if mode == "reread" else "easy",
        tags=tags,
        setup_instructions=["Create the input text files for the file-edit scenario.", "Require exact final contents."],
        config_overrides={"tools_allow_side_effect_tools": True, "planner_max_plan_steps": 9},
    )


def _reading_definition(index: int, *, mode: str) -> BenchmarkTaskDefinition:
    task_id = f"reading_generated_{mode}_{index:02d}"

    def _build(workspace: Path) -> TaskScenario:
        if mode == "structured":
            config_path = _write(workspace / f"config_{index:02d}.json", json.dumps({"service": f"svc-{index:02d}", "timeout": 10 + index}, sort_keys=True) + "\n")
            owner_path = _write(workspace / f"owner_{index:02d}.txt", f"owner=owner-{index:02d}\n")
            mode_path = _write(workspace / f"mode_{index:02d}.txt", f"mode=mode-{index % 3}\n")
            expected = {"mode": f"mode-{index % 3}", "owner": f"owner-{index:02d}", "service": f"svc-{index:02d}", "timeout": 10 + index}
            answer = stable_json_dumps(expected)
            prompt = f"Read {config_path}, {owner_path}, and {mode_path}. Return exact JSON with service, timeout, owner, and mode."
            plan = _plan_response(goal=prompt, steps=[
                _plan_step("step_read_config", "Read config", "read", expected_tool="read_text", expected_output="Config file", success_criteria="config is read."),
                _plan_step("step_read_owner", "Read owner", "read", expected_tool="read_text", expected_output="Owner file", success_criteria="owner is read.", depends_on=["step_read_config"]),
                _plan_step("step_read_mode", "Read mode", "read", expected_tool="read_text", expected_output="Mode file", success_criteria="mode is read.", depends_on=["step_read_owner"]),
                _plan_step("step_answer", "Answer user", "respond", expected_output=answer, success_criteria="The assistant returns exact JSON.", depends_on=["step_read_mode"]),
            ])
            client = _contract_scripted_client(
                plan=plan,
                tool_calls=[
                    _tool_call("read_text", {"path": config_path}),
                    _tool_call("read_text", {"path": owner_path}),
                    _tool_call("read_text", {"path": mode_path}),
                ],
                answer=answer,
            )
            contract = BenchmarkVerificationContract(task_type="reading", expected_answer=answer, expected_json=expected, expected_json_schema={"type": "object", "properties": {"service": {"type": "string"}, "timeout": {"type": "integer"}, "owner": {"type": "string"}, "mode": {"type": "string"}}, "required": ["service", "timeout", "owner", "mode"], "additionalProperties": False}, required_history_events=["file_read_requested", "verification_passed"], forbid_unexpected_workspace_changes=True)
        elif mode == "contradiction":
            left = _write(workspace / f"left_{index:02d}.txt", f"version=v{index:02d}.0\n")
            right = _write(workspace / f"right_{index:02d}.txt", f"version=v{index:02d}.1\n")
            expected = {"status": "conflict", "left": f"v{index:02d}.0", "right": f"v{index:02d}.1"}
            answer = stable_json_dumps(expected)
            prompt = f"Read {left} and {right}. Return exact JSON showing the contradiction."
            plan = _plan_response(goal=prompt, steps=[
                _plan_step("step_read_left", "Read left source", "read", expected_tool="read_text", expected_output="Left source", success_criteria="left source is read."),
                _plan_step("step_read_right", "Read right source", "read", expected_tool="read_text", expected_output="Right source", success_criteria="right source is read.", depends_on=["step_read_left"]),
                _plan_step("step_answer", "Answer user", "respond", expected_output=answer, success_criteria="The assistant returns the contradiction JSON.", depends_on=["step_read_right"]),
            ])
            client = _contract_scripted_client(
                plan=plan,
                tool_calls=[
                    _tool_call("read_text", {"path": left}),
                    _tool_call("read_text", {"path": right}),
                ],
                answer=answer,
            )
            contract = BenchmarkVerificationContract(task_type="reading", expected_answer=answer, expected_json=expected, expected_json_schema={"type": "object", "properties": {"status": {"type": "string"}, "left": {"type": "string"}, "right": {"type": "string"}}, "required": ["status", "left", "right"], "additionalProperties": False}, required_history_events=["verification_passed"], forbid_unexpected_workspace_changes=True)
        else:
            profile = _write(workspace / f"profile_{index:02d}.txt", f"name=user-{index:02d}\nteam=team-{index % 4}\n")
            expected = {"city": "unknown", "name": f"user-{index:02d}", "team": f"team-{index % 4}"}
            answer = stable_json_dumps(expected)
            prompt = f"Read only {profile}. Return exact JSON with name, team, and city. Use 'unknown' for unsupported fields."
            plan = _plan_response(goal=prompt, steps=[
                _plan_step("step_read_profile", "Read the profile", "read", expected_tool="read_text", expected_output="Profile text", success_criteria="The profile is read."),
                _plan_step("step_answer", "Answer user", "respond", expected_output=answer, success_criteria="The assistant returns JSON with null for unsupported fields.", depends_on=["step_read_profile"]),
            ])
            client = _contract_scripted_client(
                plan=plan,
                tool_calls=[_tool_call("read_text", {"path": profile})],
                answer=answer,
            )
            contract = BenchmarkVerificationContract(task_type="reading", expected_answer=answer, expected_json=expected, expected_json_schema={"type": "object", "properties": {"name": {"type": "string"}, "team": {"type": "string"}, "city": {"type": "string"}}, "required": ["name", "team", "city"], "additionalProperties": False}, required_history_events=["verification_passed"], forbid_unexpected_workspace_changes=True)
        return TaskScenario(prompt=prompt, workspace=workspace, model_client=client, verification_contract=contract)

    tags = ["reading", mode]
    if mode != "structured":
        tags.append("adversarial")
    return BenchmarkTaskDefinition(
        task_id=task_id,
        task_type="reading",
        description=f"Generated reading benchmark in {mode} mode.",
        build=_build,
        difficulty="medium" if mode != "structured" else "easy",
        tags=tags,
        setup_instructions=["Create the source files for the extraction task.", "Require exact structured output with no unsupported claims."],
        config_overrides={},
    )


def _multistep_project_definition(index: int, *, environment: bool = False) -> BenchmarkTaskDefinition:
    task_id = f"multi_step_generated_project_{index:02d}"
    topic = f"workflow_{index:02d}"
    version_value = f"{index:02d}.4"
    answer = f"synced-{index:02d}"

    def _build(workspace: Path) -> TaskScenario:
        package_dir = workspace / topic
        _write(package_dir / "__init__.py", "")
        version = _write(workspace / f"version_{index:02d}.txt", version_value + "\n")
        constants = _write(package_dir / "constants.py", 'VERSION = "0.0"\n')
        app = _write(package_dir / "app.py", f"from {topic}.constants import VERSION\n\n\ndef current_version() -> str:\n    return VERSION + '-stale'\n")
        test_file = f"test_{topic}.py"
        _write(
            workspace / test_file,
            (
                "import unittest\n\n"
                f"from {topic}.app import current_version\n"
                f"from {topic}.constants import VERSION\n\n\n"
                f"class {_class_name(topic)}SyncTests(unittest.TestCase):\n"
                f"    def test_version_constant(self) -> None:\n        self.assertEqual(VERSION, '{version_value}')\n\n"
                f"    def test_current_version(self) -> None:\n        self.assertEqual(current_version(), '{version_value}')\n\n\n"
                "if __name__ == '__main__':\n    unittest.main()\n"
            ),
        )
        read_tool = "read_file" if environment else "read_text"
        write_tool = "write_file" if environment else "edit_text"
        prompt = f"Read {version}, {constants}, and {app}. Synchronize the project, verify it, and reply {answer}."
        steps = [
            _plan_step("step_read_version", "Read version file", "read", expected_tool=read_tool, expected_output="Version text", success_criteria="The version file is read."),
            _plan_step("step_read_constants", "Read constants", "read", expected_tool=read_tool, expected_output="Constants source", success_criteria="constants.py is read.", depends_on=["step_read_version"]),
            _plan_step("step_read_app", "Read app module", "read", expected_tool=read_tool, expected_output="App source", success_criteria="app.py is read.", depends_on=["step_read_constants"]),
            _plan_step(
                "step_write_constants",
                "Write the correct version constant",
                "write",
                expected_tool=write_tool,
                expected_output="Updated constants.py",
                success_criteria="constants.py uses the target version.",
                depends_on=["step_read_app"],
                verification_checks=[
                    {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                    {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
                    {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": write_tool},
                    {"name": "file_contains_version", "check_type": "file_contains", "path": str(constants), "pattern": f'VERSION = "{version_value}"'},
                ],
                required_conditions=["dependencies_completed", "tool_result_present", "tool_name_matches", "file_contains_version"],
                optional_conditions=[],
            ),
            _plan_step(
                "step_write_app",
                "Write the matching app logic",
                "write",
                expected_tool=write_tool,
                expected_output="Updated app.py",
                success_criteria="app.py returns the synchronized version.",
                depends_on=["step_write_constants"],
                verification_checks=[
                    {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                    {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
                    {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": write_tool},
                    {"name": "file_contains_current_version", "check_type": "file_contains", "path": str(app), "pattern": "return VERSION"},
                ],
                required_conditions=["dependencies_completed", "tool_result_present", "tool_name_matches", "file_contains_current_version"],
                optional_conditions=[],
            ),
            _plan_step(
                "step_run_tests",
                "Run unittest suite",
                "tool",
                expected_tool="run_tests",
                expected_output="Passing unittest result",
                success_criteria="The project tests pass.",
                depends_on=["step_write_app"],
                verification_checks=[
                    {"name": "dependencies_completed", "check_type": "dependencies_completed"},
                    {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"},
                    {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "run_tests"},
                    {"name": "tests_pass", "check_type": "exact_match", "actual_source": "tool_output.passed", "expected": True},
                ],
                required_conditions=["dependencies_completed", "tool_result_present", "tool_name_matches", "tests_pass"],
                optional_conditions=[],
            ),
            _plan_step("step_answer", "Answer the user", "respond", expected_output=answer, success_criteria=f"The assistant replies {answer}.", depends_on=["step_run_tests"]),
        ]
        responses = [
            _plan_response(goal=prompt, steps=steps),
            _tool_call(read_tool, {"path": str(version)}),
            _tool_call(read_tool, {"path": str(constants)}),
            _tool_call(read_tool, {"path": str(app)}),
        ]
        if environment:
            responses.extend([
                _tool_call("write_file", {"path": str(constants), "content": f'VERSION = "{version_value}"\n'}),
                _tool_call("write_file", {"path": str(app), "content": f"from {topic}.constants import VERSION\n\n\ndef current_version() -> str:\n    return VERSION\n"}),
            ])
        else:
            responses.extend([
                _tool_call("edit_text", {"path": str(constants), "operation": "replace_pattern_once", "pattern": 'VERSION = "0.0"', "replacement": f'VERSION = "{version_value}"'}),
                _tool_call("edit_text", {"path": str(app), "operation": "replace_pattern_once", "pattern": "return VERSION + '-stale'", "replacement": "return VERSION"}),
            ])
        responses.extend([
            _tool_call("run_tests", {"command": ["python3", "-m", "unittest", "-q", test_file]}),
            answer,
        ])
        return TaskScenario(
            prompt=prompt,
            workspace=workspace,
            model_client=ScriptedBenchmarkClient(responses=responses),
            verification_contract=BenchmarkVerificationContract(
                task_type="multi_step",
                expected_answer=answer,
                expected_files={str(constants): f'VERSION = "{version_value}"\n', str(app): f"from {topic}.constants import VERSION\n\n\ndef current_version() -> str:\n    return VERSION\n"},
                command=["python3", "-m", "unittest", "-q", test_file],
                command_cwd=str(workspace),
                command_framework="unittest",
                required_tools_used=[read_tool, write_tool, "run_tests"],
                min_tool_calls=6,
                required_history_events=["process_completed", "verification_passed"],
                forbid_unexpected_workspace_changes=True,
            ),
        )

    return BenchmarkTaskDefinition(
        task_id=task_id,
        task_type="multi_step",
        description=f"Synchronize a realistic multi-file project through a full read/edit/test loop ({'environment' if environment else 'editor'} mode).",
        build=_build,
        difficulty="hard" if environment else "medium",
        tags=["multi-step", "multifile", "realistic-code", "project-consistency", *( ["environment"] if environment else ["editor"] )],
        setup_instructions=[
            "Create a version file and a two-module package with stale values.",
            "Create a unittest file that verifies project-wide consistency.",
            "Require the agent to read, edit, test, and respond.",
        ],
        config_overrides={"tools_allow_side_effect_tools": True, "planner_max_plan_steps": 9},
    )


def _multi_step_long_run_definition(index: int) -> BenchmarkTaskDefinition:
    task_id = f"multi_step_long_run_{index:02d}"
    topic = f"recovery_{index:02d}"
    version_value = f"{index:02d}.9"
    answer = f"recovered-{index:02d}"

    def _build(workspace: Path) -> TaskScenario:
        package_dir = workspace / topic
        _write(package_dir / "__init__.py", "")
        version = _write(workspace / f"version_{index:02d}.txt", version_value + "\n")
        constants = _write(package_dir / "constants.py", 'VERSION = "broken"\n')
        app = _write(package_dir / "app.py", f"from {topic}.constants import VERSION\n\n\ndef current_version() -> str:\n    return VERSION + '-old'\n")
        report = _write(package_dir / "report.py", f"from {topic}.app import current_version\n\n\ndef render() -> str:\n    return 'pending'\n")
        test_file = f"test_{topic}.py"
        _write(
            workspace / test_file,
            (
                "import unittest\n\n"
                f"from {topic}.constants import VERSION\n"
                f"from {topic}.app import current_version\n"
                f"from {topic}.report import render\n\n\n"
                f"class {_class_name(topic)}RecoveryTests(unittest.TestCase):\n"
                f"    def test_version_constant(self) -> None:\n        self.assertEqual(VERSION, '{version_value}')\n\n"
                f"    def test_current_version(self) -> None:\n        self.assertEqual(current_version(), '{version_value}')\n\n"
                f"    def test_report(self) -> None:\n        self.assertEqual(render(), 'version={version_value}')\n\n\n"
                "if __name__ == '__main__':\n    unittest.main()\n"
            ),
        )
        plan1 = _plan_response(
            goal=f"Read {version}, {constants}, {app}, and {report}. Recover the project, verify it, and reply {answer}.",
            steps=[
                _plan_step("step_read_version", "Read version", "read", expected_tool="read_text", expected_output="Version text", success_criteria="version file is read."),
                _plan_step("step_read_constants", "Read constants", "read", expected_tool="read_text", expected_output="Constants source", success_criteria="constants.py is read.", depends_on=["step_read_version"]),
                _plan_step("step_read_app", "Read app", "read", expected_tool="read_text", expected_output="App source", success_criteria="app.py is read.", depends_on=["step_read_constants"]),
                _plan_step("step_read_report", "Read report", "read", expected_tool="read_text", expected_output="Report source", success_criteria="report.py is read.", depends_on=["step_read_app"]),
                _plan_step("step_fix_placeholder", "Fix the project", "write", expected_tool="edit_text", expected_output="Updated sources", success_criteria="The project is updated.", depends_on=["step_read_report"]),
                _plan_step("step_answer_placeholder", "Answer the user", "respond", expected_output=answer, success_criteria=f"The assistant replies {answer}.", depends_on=["step_fix_placeholder"]),
            ],
        )
        plan2 = _plan_response(
            goal=f"Read {version}, {constants}, {app}, and {report}. Recover the project, verify it, and reply {answer}.",
            steps=[
                _plan_step("step_read_report_replan", "Read report", "read", expected_tool="read_text", expected_output="Report source", success_criteria="report.py is read."),
                _plan_step("step_fix_constants_replan", "Fix constants", "write", expected_tool="edit_text", expected_output="Updated constants.py", success_criteria="constants.py is fixed.", depends_on=["step_read_report_replan"], verification_checks=[{"name": "dependencies_completed", "check_type": "dependencies_completed"}, {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"}, {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"}, {"name": "file_contains_version", "check_type": "file_contains", "path": str(constants), "pattern": f'VERSION = "{version_value}"'}], required_conditions=["dependencies_completed", "tool_result_present", "tool_name_matches", "file_contains_version"], optional_conditions=[]),
                _plan_step("step_fix_app_replan", "Fix app", "write", expected_tool="edit_text", expected_output="Updated app.py", success_criteria="app.py returns VERSION.", depends_on=["step_fix_constants_replan"], verification_checks=[{"name": "dependencies_completed", "check_type": "dependencies_completed"}, {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"}, {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"}, {"name": "file_contains_return", "check_type": "file_contains", "path": str(app), "pattern": "return VERSION"}], required_conditions=["dependencies_completed", "tool_result_present", "tool_name_matches", "file_contains_return"], optional_conditions=[]),
                _plan_step("step_answer_replan_placeholder", "Answer the user", "respond", expected_output=answer, success_criteria=f"The assistant replies {answer}.", depends_on=["step_fix_app_replan"]),
            ],
        )
        plan3 = _plan_response(
            goal=f"Read {version}, {constants}, {app}, and {report}. Recover the project, verify it, and reply {answer}.",
            steps=[
                _plan_step("step_fix_app_final", "Fix app", "write", expected_tool="edit_text", expected_output="Updated app.py", success_criteria="app.py returns VERSION.", verification_checks=[{"name": "dependencies_completed", "check_type": "dependencies_completed"}, {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"}, {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"}, {"name": "file_contains_return", "check_type": "file_contains", "path": str(app), "pattern": "return VERSION"}], required_conditions=["dependencies_completed", "tool_result_present", "tool_name_matches", "file_contains_return"], optional_conditions=[]),
                _plan_step("step_fix_report_final", "Fix report", "write", expected_tool="edit_text", expected_output="Updated report.py", success_criteria="report.py renders the correct version.", depends_on=["step_fix_app_final"], verification_checks=[{"name": "dependencies_completed", "check_type": "dependencies_completed"}, {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"}, {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"}, {"name": "file_contains_render", "check_type": "file_contains", "path": str(report), "pattern": f"version={version_value}"}], required_conditions=["dependencies_completed", "tool_result_present", "tool_name_matches", "file_contains_render"], optional_conditions=[]),
                _plan_step("step_run_tests_final", "Run tests", "tool", expected_tool="run_tests", expected_output="Passing tests", success_criteria="Tests pass.", depends_on=["step_fix_report_final"], verification_checks=[{"name": "dependencies_completed", "check_type": "dependencies_completed"}, {"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"}, {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "run_tests"}, {"name": "tests_pass", "check_type": "exact_match", "actual_source": "tool_output.passed", "expected": True}], required_conditions=["dependencies_completed", "tool_result_present", "tool_name_matches", "tests_pass"], optional_conditions=[]),
                _plan_step("step_read_report_final", "Reread report", "read", expected_tool="read_text", expected_output="Report source", success_criteria="report.py is reread.", depends_on=["step_run_tests_final"]),
                _plan_step("step_read_app_final", "Reread app", "read", expected_tool="read_text", expected_output="App source", success_criteria="app.py is reread.", depends_on=["step_read_report_final"]),
                _plan_step("step_answer_final", "Answer the user", "respond", expected_output=answer, success_criteria=f"The assistant replies {answer}.", depends_on=["step_read_app_final"]),
            ],
        )
        prompt = f"Read {version}, {constants}, {app}, and {report}. Recover the project, verify it, and reply {answer}."
        responses = [
            plan1,
            _tool_call("read_text", {"path": str(version)}),
            _tool_call("read_text", {"path": str(constants)}),
            _tool_call("read_text", {"path": str(app)}),
            _tool_call("echo", {"text": "wrong"}),
            plan2,
            _tool_call("read_text", {"path": str(report)}),
            _tool_call("edit_text", {"path": str(constants), "operation": "replace_pattern_once", "pattern": 'VERSION = "broken"', "replacement": f'VERSION = "{version_value}"'}),
            _tool_call("echo", {"text": "wrong"}),
            plan3,
            _tool_call("edit_text", {"path": str(app), "operation": "replace_pattern_once", "pattern": "return VERSION + '-old'", "replacement": "return VERSION"}),
            _tool_call("edit_text", {"path": str(report), "operation": "replace_pattern_once", "pattern": "return 'pending'", "replacement": f"return 'version={version_value}'"}),
            _tool_call("run_tests", {"command": ["python3", "-m", "unittest", "-q", test_file]}),
            _tool_call("read_text", {"path": str(report)}),
            _tool_call("read_text", {"path": str(app)}),
            answer,
        ]
        return TaskScenario(
            prompt=prompt,
            workspace=workspace,
            model_client=ScriptedBenchmarkClient(responses=responses),
            verification_contract=BenchmarkVerificationContract(
                task_type="multi_step",
                expected_answer=answer,
                expected_files={
                    str(constants): f'VERSION = "{version_value}"\n',
                    str(app): f"from {topic}.constants import VERSION\n\n\ndef current_version() -> str:\n    return VERSION\n",
                    str(report): f"from {topic}.app import current_version\n\n\ndef render() -> str:\n    return 'version={version_value}'\n",
                },
                command=["python3", "-m", "unittest", "-q", test_file],
                command_cwd=str(workspace),
                command_framework="unittest",
                required_history_events=["replan_triggered", "step_failed", "verification_passed"],
                required_event_counts={"replan_triggered": 2, "step_started": 10, "step_failed": 2},
                required_tools_used=["read_text", "edit_text", "run_tests"],
                min_tool_calls=10,
                forbid_unexpected_workspace_changes=True,
            ),
        )

    return BenchmarkTaskDefinition(
        task_id=task_id,
        task_type="multi_step",
        description="Long-run recovery benchmark with multiple replans and a recovered final result.",
        build=_build,
        difficulty="hard",
        tags=["multi-step", "long-run", "recovery", "multifile", "realistic-code", "project-consistency"],
        setup_instructions=[
            "Create a four-file project with stale constants and formatting.",
            "Script two failed branches that require replanning before the final successful path.",
            "Require at least ten started steps and two replans.",
        ],
        config_overrides={
            "tools_allow_side_effect_tools": True,
            "planner_max_replans": 2,
            "planner_max_plan_steps": 12,
            "runtime_max_reasoning_steps": 14,
            "runtime_max_total_actions": 16,
            "runtime_max_tool_steps": 12,
            "runtime_tool_call_budget": 14,
        },
    )


def _failure_definition(index: int, *, mode: str) -> BenchmarkTaskDefinition:
    task_id = f"failure_generated_{mode}_{index:02d}"

    def _build(workspace: Path) -> TaskScenario:
        if mode == "wrong_tool":
            payload = _write(workspace / f"payload_{index:02d}.txt", f"value={40 + index}\n")
            prompt = f"Read {payload} and reply with the value."
            plan = _plan_response(goal=prompt, steps=[_plan_step("step_read", "Read payload", "read", expected_tool="read_text", expected_output="Payload", success_criteria="Payload is read."), _plan_step("step_answer", "Answer", "respond", expected_output=str(40 + index), success_criteria="The assistant replies with the value.", depends_on=["step_read"] )])
            wrong_tool = "calculator" if index % 2 == 0 else "echo"
            wrong_input = {"expression": f"{20 + index} + {20}"} if wrong_tool == "calculator" else {"text": "wrong"}
            responses = [plan, _tool_call(wrong_tool, wrong_input)]
            contract = BenchmarkVerificationContract(task_type="failure", required_history_events=["tool_graph_rejected", "verification_failed"])
            expected_failure_category = "wrong_tool_usage"
        elif mode == "bad_plan":
            target = _write(workspace / f"variant_{index:02d}.txt", "status=broken\n")
            prompt = f"Inspect {target}. Summarize the status in one word."
            invalid_plan = json.dumps({"goal": prompt, "success_criteria": "done", "fallback_strategy": "retry", "steps": [{"step_id": f"broken_{index:02d}"}]})
            responses = [invalid_plan]
            contract = BenchmarkVerificationContract(
                task_type="failure",
                required_history_events=["error", "reasoning_completed"],
                expected_stop_reason="plan_generation_failed",
            )
            expected_failure_category = "bad_planning"
        else:
            document = _write(workspace / f"loop_{index:02d}.txt", "alpha\nbeta\n")
            prompt = f"Read {document}, update beta to gamma, and reply updated-{index:02d}."
            plan = _plan_response(goal=prompt, steps=[_plan_step("step_read", "Read document", "read", expected_tool="read_text", expected_output="Document", success_criteria="Document is read."), _plan_step("step_fix", "Fix document", "write", expected_tool="edit_text", expected_output="Updated document", success_criteria="The document is updated.", depends_on=["step_read"], verification_checks=[{"name": "tool_result_present", "check_type": "artifact_present", "artifact": "tool_result"}, {"name": "tool_name_matches", "check_type": "tool_name_equals", "expected": "edit_text"}, {"name": "file_contains_gamma", "check_type": "file_contains", "path": str(document), "pattern": "gamma"}], required_conditions=["tool_result_present", "tool_name_matches", "file_contains_gamma"], optional_conditions=[]), _plan_step("step_answer", "Answer", "respond", expected_output=f"updated-{index:02d}", success_criteria="The assistant replies updated.", depends_on=["step_fix"])])
            responses = [
                plan,
                _tool_call("read_text", {"path": str(document)}),
                _tool_call("read_text", {"path": str(document)}),
                _tool_call("read_text", {"path": str(document)}),
                "not done",
            ]
            contract = BenchmarkVerificationContract(task_type="failure", required_history_events=["duplicate_action_detected"], required_event_counts={"tool_called": 2})
            expected_failure_category = "loop_no_progress"
        return TaskScenario(prompt=prompt, workspace=workspace, model_client=ScriptedBenchmarkClient(responses=responses), verification_contract=contract, expected_outcome="expected_failure", expected_failure_category=expected_failure_category)

    return BenchmarkTaskDefinition(
        task_id=task_id,
        task_type="failure",
        description=f"Generated adversarial failure benchmark in {mode} mode.",
        build=_build,
        difficulty="hard" if mode == "repeated_action" else "medium",
        tags=["failure", mode, "adversarial", *(["false-positive-killer"] if mode in {"bad_plan", "repeated_action"} else [])],
        setup_instructions=["Create the minimal files for the failure scenario.", "Require the runtime to fail safely with the expected classification."],
        config_overrides={"tools_allow_side_effect_tools": True, "planner_max_replans": 0 if mode != "bad_plan" else 0, "runtime_max_tool_steps": 3},
    )


def _quality_definition(index: int, *, mode: str) -> BenchmarkTaskDefinition:
    task_id = f"quality_generated_{mode}_{index:02d}"

    def _build(workspace: Path) -> TaskScenario:
        if mode == "vague":
            prompt = "make a game"
            answer = "scoped"
            expanded_goal = "make a game Build a small arcade prototype with one core mechanic and a playable loop."
            plan = _plan_response(goal=expanded_goal, steps=[_plan_step("step_answer", "Answer the user", "respond", expected_output=answer, success_criteria=f"The assistant replies {answer}.")])
            client = _contract_scripted_client(plan=plan, answer=answer)
            contract = BenchmarkVerificationContract(task_type="quality", expected_answer=answer, required_history_events=["task_expanded", "verification_passed"])
            oracle = PromptUnderstandingOracle(task_type="vague", completeness="incomplete", requires_expansion=True, requires_decomposition=False, expand_task=True, split_task=False, ask_user=False, assume_missing=True, generate_ideas=True, strategy_profile="generic", detected_goals_contains=["make"])
        elif mode == "decomposed":
            facts = _write(workspace / f"facts_{index:02d}.txt", f"owner=owner-{index:02d}\nstatus=ready\n")
            answer = stable_json_dumps({"owner": f"owner-{index:02d}", "status": "ready"})
            prompt = f"1. Read {facts}\n2. Return exact JSON summary"
            plan = _plan_response(goal=prompt, steps=[_plan_step("step_read", "Read facts", "read", expected_tool="read_text", expected_output="facts", success_criteria="facts are read."), _plan_step("step_answer", "Answer", "respond", expected_output=answer, success_criteria="The assistant returns exact JSON.", depends_on=["step_read"])])
            client = _contract_scripted_client(
                plan=plan,
                tool_calls=[_tool_call("read_text", {"path": str(facts)})],
                answer=answer,
            )
            contract = BenchmarkVerificationContract(task_type="quality", expected_answer=answer, expected_json={"owner": f"owner-{index:02d}", "status": "ready"}, expected_json_schema={"type": "object", "properties": {"owner": {"type": "string"}, "status": {"type": "string"}}, "required": ["owner", "status"], "additionalProperties": False}, required_history_events=["prompt_analyzed", "decision_made", "verification_passed"])
            oracle = PromptUnderstandingOracle(task_type="already_decomposed", completeness="complete", requires_expansion=False, requires_decomposition=True, expand_task=False, split_task=True, ask_user=False, strategy_profile="reading", detected_entities_contains=[f"facts_{index:02d}.txt"])
        elif mode == "incomplete":
            prompt = f"Can you handle request {index:02d}?"
            client = ScriptedBenchmarkClient(responses=[])
            contract = BenchmarkVerificationContract(task_type="quality", expected_answer_contains=["I need clarification before I can continue."], required_history_events=["prompt_analyzed", "decision_made", "reasoning_completed"], expected_stop_reason="prompt_incomplete")
            oracle = PromptUnderstandingOracle(task_type="incomplete", completeness="incomplete", requires_expansion=True, requires_decomposition=False, expand_task=True, split_task=False, ask_user=True, assume_missing=False, generate_ideas=False, strategy_profile="generic")
        else:
            debug_log = _write(workspace / f"bench-debug-{index:02d}.log", f"ERROR code={100 + index}\nWARNING slow=true\n")
            answer = stable_json_dumps({"error_code": 100 + index, "warning": True})
            prompt = f"Read {debug_log}. Return exact JSON with error_code and warning."
            plan = _plan_response(goal=prompt, steps=[_plan_step("step_read", "Read debug log", "read", expected_tool="read_text", expected_output="Log text", success_criteria="The debug log is read."), _plan_step("step_answer", "Answer", "respond", expected_output=answer, success_criteria="The assistant returns exact JSON.", depends_on=["step_read"])])
            client = _contract_scripted_client(
                plan=plan,
                tool_calls=[_tool_call("read_text", {"path": str(debug_log)})],
                answer=answer,
            )
            contract = BenchmarkVerificationContract(task_type="quality", expected_answer=answer, expected_json={"error_code": 100 + index, "warning": True}, expected_json_schema={"type": "object", "properties": {"error_code": {"type": "integer"}, "warning": {"type": "boolean"}}, "required": ["error_code", "warning"], "additionalProperties": False}, required_history_events=["verification_passed"])
            oracle = PromptUnderstandingOracle(task_type="structured", completeness="complete", expand_task=False, split_task=True, strategy_profile="reading", detected_entities_contains=[f"bench-debug-{index:02d}.log"], detected_goals_contains=["read"])
        return TaskScenario(prompt=prompt, workspace=workspace, model_client=client, verification_contract=contract, oracle=oracle)

    return BenchmarkTaskDefinition(
        task_id=task_id,
        task_type="quality",
        description=f"Generated prompt-understanding benchmark in {mode} mode.",
        build=_build,
        difficulty="medium" if mode in {"decomposed", "debug_reading"} else "easy",
        tags=["quality", mode, "prompt-understanding", *(["false-positive-killer"] if mode in {"incomplete", "debug_reading"} else [])],
        setup_instructions=["Prepare the prompt-understanding scenario inputs.", "Require the runtime to follow the quality oracle exactly."],
        config_overrides={},
    )


def generated_benchmark_tasks() -> list[BenchmarkTaskDefinition]:
    tasks: list[BenchmarkTaskDefinition] = []
    tasks.extend(_coding_multifile_definition(index) for index in range(1, 25))
    tasks.extend(_coding_multifile_definition(index, environment=True) for index in range(25, 41))

    tasks.extend(_file_edit_definition(index, mode="exact") for index in range(1, 9))
    tasks.extend(_file_edit_definition(index, mode="replace_all") for index in range(9, 17))
    tasks.extend(_file_edit_definition(index, mode="reread") for index in range(17, 26))

    tasks.extend(_reading_definition(index, mode="structured") for index in range(1, 11))
    tasks.extend(_reading_definition(index, mode="contradiction") for index in range(11, 18))
    tasks.extend(_reading_definition(index, mode="hallucination_guard") for index in range(18, 26))

    tasks.extend(_multistep_project_definition(index) for index in range(1, 15))
    tasks.extend(_multistep_project_definition(index, environment=True) for index in range(15, 29))
    tasks.extend(_multi_step_long_run_definition(index) for index in range(23, 25))

    tasks.extend(_failure_definition(index, mode="wrong_tool") for index in range(1, 11))
    tasks.extend(_failure_definition(index, mode="bad_plan") for index in range(11, 19))
    tasks.extend(_failure_definition(index, mode="repeated_action") for index in range(19, 31))

    tasks.extend(_quality_definition(index, mode="vague") for index in range(1, 7))
    tasks.extend(_quality_definition(index, mode="decomposed") for index in range(7, 13))
    tasks.extend(_quality_definition(index, mode="incomplete") for index in range(13, 17))
    tasks.extend(_quality_definition(index, mode="debug_reading") for index in range(17, 21))
    return tasks


def _live_coding_definition(index: int) -> BenchmarkTaskDefinition:
    task_id = f"live_coding_fix_{index:02d}"
    answer = f"fixed-live-coding-{index:02d}"

    def _scenario(workspace: Path, *, scripted: bool) -> TaskScenario:
        if index <= 2:
            function_name = f"compute_{index:02d}"
            expected = 30 + index
            source = _write(
                workspace / f"{function_name}.py",
                f"def {function_name}(a: int, b: int) -> int:\n    return 0\n",
            )
            test_name = f"test_{function_name}.py"
            _write(
                workspace / test_name,
                (
                    "import unittest\n\n"
                    f"from {function_name} import {function_name}\n\n\n"
                    f"class {_class_name(function_name)}Tests(unittest.TestCase):\n"
                    f"    def test_value(self) -> None:\n        self.assertEqual({function_name}({expected - index}, {index}), {expected})\n\n\n"
                    "if __name__ == '__main__':\n    unittest.main()\n"
                ),
            )
            fixed_files = {source: f"def {function_name}(a: int, b: int) -> int:\n    return a + b\n"}
            prompt = (
                f"Read {Path(source).name}. Replace `return 0` with `return a + b`. "
                f"Run tests in {test_name}. Reply exactly {answer}."
            )
            contract = BenchmarkVerificationContract(
                task_type="coding",
                expected_answer=answer,
                expected_files=fixed_files,
                command=["python3", "-m", "unittest", "-q", test_name],
                command_cwd=str(workspace),
                command_framework="unittest",
                required_history_events=["tool_called", "tool_result", "verification_passed"],
                required_tools_used=["read_text", "edit_text"],
                min_tool_calls=2,
                forbid_unexpected_workspace_changes=True,
            )
            if not scripted:
                return TaskScenario(prompt=prompt, workspace=workspace, model_client=None, verification_contract=contract)
            plan = _plan_response(
                goal=prompt,
                steps=[
                    _plan_step("step_read", "Read implementation", "read", expected_tool="read_text", expected_output="source", success_criteria="source read"),
                    _plan_step("step_edit", "Fix implementation", "write", expected_tool="edit_text", expected_output="updated source", success_criteria="source fixed", depends_on=["step_read"]),
                    _plan_step("step_tests", "Run tests", "tool", expected_tool="run_tests", expected_output="passing tests", success_criteria="tests pass", depends_on=["step_edit"]),
                    _plan_step("step_answer", "Answer", "respond", expected_output=answer, success_criteria="reply exact answer", depends_on=["step_tests"]),
                ],
            )
            client = ScriptedBenchmarkClient(
                responses=[
                    plan,
                    _tool_call("read_text", {"path": source}),
                    _tool_call("edit_text", {"path": source, "operation": "replace_pattern_once", "pattern": "return 0", "replacement": "return a + b"}),
                    _tool_call("run_tests", {"command": ["python3", "-m", "unittest", "-q", test_name]}),
                    answer,
                ]
            )
            return TaskScenario(prompt=prompt, workspace=workspace, model_client=client, verification_contract=contract)

        package = workspace / f"pkg_live_{index:02d}"
        module_name = package.name
        label = f"result-{index:02d}"
        base_value = 18 + index
        delta = 6 + index
        total = base_value + delta
        wrong_base = base_value - 2
        wrong_delta = delta - 1

        _write(package / "__init__.py", "")
        core = _write(package / "core.py", f"def base_value() -> int:\n    return {wrong_base}\n")
        calc = _write(
            package / "calc.py",
            f"from {module_name}.core import base_value\n\n\ndef total() -> int:\n    return base_value() + {wrong_delta}\n",
        )
        report = _write(
            package / "report.py",
            f"from {module_name}.calc import total\n\n\ndef describe() -> str:\n    return f\"{label}={{total() - 1}}\"\n",
        )
        test_name = f"test_{module_name}.py"
        _write(
            workspace / test_name,
            (
                "import unittest\n\n"
                f"from {module_name}.core import base_value\n"
                f"from {module_name}.calc import total\n"
                f"from {module_name}.report import describe\n\n\n"
                f"class {_class_name(module_name)}Tests(unittest.TestCase):\n"
                f"    def test_base_value(self) -> None:\n        self.assertEqual(base_value(), {base_value})\n\n"
                f"    def test_total(self) -> None:\n        self.assertEqual(total(), {total})\n\n"
                f"    def test_report(self) -> None:\n        self.assertEqual(describe(), \"{label}={total}\")\n\n\n"
                "if __name__ == '__main__':\n    unittest.main()\n"
            ),
        )
        fixed_core = f"def base_value() -> int:\n    return {base_value}\n"
        fixed_calc = f"from {module_name}.core import base_value\n\n\ndef total() -> int:\n    return base_value() + {delta}\n"
        fixed_report = f"from {module_name}.calc import total\n\n\ndef describe() -> str:\n    return f\"{label}={{total()}}\"\n"
        prompt = (
            f"Read {core}, {calc}, and {report}. Fix all three files so tests in {test_name} pass "
            f"and the modules stay consistent. Reply exactly {answer}."
        )
        contract = BenchmarkVerificationContract(
            task_type="coding",
            expected_answer=answer,
            expected_files={core: fixed_core, calc: fixed_calc, report: fixed_report},
            command=["python3", "-m", "unittest", "-q", test_name],
            command_cwd=str(workspace),
            command_framework="unittest",
            required_history_events=["tool_called", "tool_result", "verification_passed"],
            required_tools_used=["read_text", "edit_text", "run_tests"],
            min_tool_calls=7,
            forbid_unexpected_workspace_changes=True,
        )
        if not scripted:
            return TaskScenario(prompt=prompt, workspace=workspace, model_client=None, verification_contract=contract)
        plan = _plan_response(
            goal=prompt,
            steps=[
                _plan_step("step_read_core", "Read core", "read", expected_tool="read_text", expected_output="core", success_criteria="core read"),
                _plan_step("step_read_calc", "Read calc", "read", expected_tool="read_text", expected_output="calc", success_criteria="calc read", depends_on=["step_read_core"]),
                _plan_step("step_read_report", "Read report", "read", expected_tool="read_text", expected_output="report", success_criteria="report read", depends_on=["step_read_calc"]),
                _plan_step("step_fix_core", "Fix core", "write", expected_tool="edit_text", expected_output="fixed core", success_criteria="core fixed", depends_on=["step_read_report"]),
                _plan_step("step_fix_calc", "Fix calc", "write", expected_tool="edit_text", expected_output="fixed calc", success_criteria="calc fixed", depends_on=["step_fix_core"]),
                _plan_step("step_fix_report", "Fix report", "write", expected_tool="edit_text", expected_output="fixed report", success_criteria="report fixed", depends_on=["step_fix_calc"]),
                _plan_step("step_tests", "Run tests", "tool", expected_tool="run_tests", expected_output="passing tests", success_criteria="tests pass", depends_on=["step_fix_report"]),
                _plan_step("step_answer", "Answer", "respond", expected_output=answer, success_criteria="reply exact answer", depends_on=["step_tests"]),
            ],
        )
        client = ScriptedBenchmarkClient(
            responses=[
                plan,
                _tool_call("read_text", {"path": str(core)}),
                _tool_call("read_text", {"path": str(calc)}),
                _tool_call("read_text", {"path": str(report)}),
                _tool_call("edit_text", {"path": str(core), "operation": "replace_pattern_once", "pattern": f"return {wrong_base}", "replacement": f"return {base_value}"}),
                _tool_call("edit_text", {"path": str(calc), "operation": "replace_pattern_once", "pattern": f"return base_value() + {wrong_delta}", "replacement": f"return base_value() + {delta}"}),
                _tool_call("edit_text", {"path": str(report), "operation": "replace_pattern_once", "pattern": f'return f\"{label}={{total() - 1}}\"', "replacement": f'return f\"{label}={{total()}}\"'}),
                _tool_call("run_tests", {"command": ["python3", "-m", "unittest", "-q", test_name]}),
                answer,
            ]
        )
        return TaskScenario(prompt=prompt, workspace=workspace, model_client=client, verification_contract=contract)

    return BenchmarkTaskDefinition(
        task_id=task_id,
        task_type="coding",
        description="Live benchmark coding fix with deterministic unittest verification.",
        build=lambda workspace: _scenario(workspace, scripted=True),
        build_live=lambda workspace: _scenario(workspace, scripted=False),
        difficulty="easy" if index == 1 else ("medium" if index in {2, 3} else "hard"),
        tags=[
            "live-subset",
            "coding",
            "environment",
            "run-tests",
            "verification-edge",
            *(["multifile", "realistic-code", "project-consistency"] if index >= 3 else ["single-file"]),
        ],
        setup_instructions=[
            "Create Python source files with deterministic failing tests.",
            "Require exact file fixes and execution-based verification.",
            *(["Use three coordinated source files for the harder live coding tasks."] if index >= 3 else []),
        ],
        config_overrides={
            "tools_allow_side_effect_tools": True,
            "runtime_max_tool_steps": 10 if index >= 3 else 6,
            "runtime_tool_call_budget": 10 if index >= 3 else 6,
            "runtime_max_reasoning_steps": 14 if index >= 3 else 8,
            "runtime_max_total_actions": 14 if index >= 3 else 12,
        },
    )


def _live_file_edit_definition(index: int) -> BenchmarkTaskDefinition:
    task_id = f"live_file_edit_{index:02d}"
    answer = f"edited-live-{index:02d}"

    def _scenario(workspace: Path, *, scripted: bool) -> TaskScenario:
        path = workspace / f"note_{index:02d}.txt"
        if index == 1:
            original = f"hello-{index:02d}\n"
            fixed = f"world-{index:02d}\n"
            edit_call = _tool_call("edit_text", {"path": str(path), "operation": "replace_range", "start": 0, "end": len(original), "replacement": fixed})
            prompt = f"Edit {path.name} so the file content becomes exactly world-{index:02d} followed by a newline. Reply exactly {answer}."
            min_tool_calls = 2
        elif index == 2:
            original = f"title=note-{index:02d}\nstatus=draft\n"
            fixed = f"title=note-{index:02d}\nstatus=ready\n"
            edit_call = _tool_call("edit_text", {"path": str(path), "operation": "replace_pattern_once", "pattern": "status=draft", "replacement": "status=ready"})
            prompt = f"Read {path.name}. Replace only status=draft with status=ready. Reply exactly {answer}."
            min_tool_calls = 2
        elif index == 3:
            original = f"alpha={index}\nbeta={index + 1}\n"
            fixed = f"alpha={index}\nbeta={index + 10}\n"
            edit_call = _tool_call("edit_text", {"path": str(path), "operation": "replace_pattern_once", "pattern": f"beta={index + 1}", "replacement": f"beta={index + 10}"})
            prompt = f"Read {path.name}. Change only beta to {index + 10}, then verify by rereading the file. Reply exactly {answer}."
            min_tool_calls = 3
        elif index == 4:
            original = "color=red\nshape=red\nstatus=draft\n"
            fixed = "color=blue\nshape=blue\nstatus=ready\n"
            edit_call = _tool_call("edit_text", {"path": str(path), "operation": "replace_pattern_all", "pattern": "red", "replacement": "blue"})
            prompt = f"Read {path.name}. Replace both red values with blue and change status=draft to status=ready. Reply exactly {answer}."
            min_tool_calls = 3
        else:
            original = "value=41\n"
            fixed = "value=42\n"
            edit_call = _tool_call("edit_text", {"path": str(path), "operation": "replace_pattern_once", "pattern": "value=41", "replacement": "value=42"})
            prompt = f"Read {path.name}. Make the file content exactly value=42 followed by a newline. Extra spaces or a missing newline are wrong. Reply exactly {answer}."
            min_tool_calls = 2

        _write(path, original)
        contract = BenchmarkVerificationContract(
            task_type="file_edit",
            expected_answer=answer,
            expected_files={path: fixed},
            required_history_events=["tool_called", "tool_result", "verification_passed"],
            required_tools_used=["read_text", "edit_text"],
            min_tool_calls=min_tool_calls,
            forbid_unexpected_workspace_changes=True,
        )
        if not scripted:
            return TaskScenario(prompt=prompt, workspace=workspace, model_client=None, verification_contract=contract)
        responses = [
            _plan_response(
                goal=prompt,
                steps=[
                    _plan_step("step_read", "Read file", "read", expected_tool="read_text", expected_output="file text", success_criteria="file read"),
                    _plan_step("step_edit", "Edit file", "write", expected_tool="edit_text", expected_output="updated file", success_criteria="file updated", depends_on=["step_read"]),
                    *(
                        [_plan_step("step_reread", "Verify file", "read", expected_tool="read_text", expected_output="updated file", success_criteria="updated file reread", depends_on=["step_edit"])]
                        if index in {3, 4}
                        else []
                    ),
                    _plan_step("step_answer", "Answer", "respond", expected_output=answer, success_criteria="reply exact answer", depends_on=["step_reread" if index in {3, 4} else "step_edit"]),
                ],
            ),
            _tool_call("read_text", {"path": str(path)}),
            edit_call,
        ]
        if index in {3, 4}:
            responses.append(_tool_call("read_text", {"path": str(path)}))
        if index == 4:
            responses.insert(
                3,
                _tool_call("edit_text", {"path": str(path), "operation": "replace_pattern_once", "pattern": "status=draft", "replacement": "status=ready"}),
            )
        responses.append(answer)
        client = ScriptedBenchmarkClient(responses=responses)
        return TaskScenario(prompt=prompt, workspace=workspace, model_client=client, verification_contract=contract)

    return BenchmarkTaskDefinition(
        task_id=task_id,
        task_type="file_edit",
        description="Live benchmark exact file edit with deterministic content verification.",
        build=lambda workspace: _scenario(workspace, scripted=True),
        build_live=lambda workspace: _scenario(workspace, scripted=False),
        difficulty="easy" if index == 1 else ("medium" if index in {2, 3} else "hard"),
        tags=[
            "live-subset",
            "file-edit",
            "environment",
            "filesystem",
            *(["verification-edge"] if index in {4, 5} else []),
            *(["long-run"] if index in {3, 4} else []),
        ],
        setup_instructions=[
            "Create a deterministic text-edit task with exact file-state verification.",
            *(["Require a reread after modification for the longer file-edit tasks."] if index in {3, 4} else []),
        ],
        config_overrides={"tools_allow_side_effect_tools": True},
    )


def _live_reading_definition(index: int) -> BenchmarkTaskDefinition:
    task_id = f"live_reading_{index:02d}"

    def _scenario(workspace: Path, *, scripted: bool) -> TaskScenario:
        required_tools = ["read_text"]
        min_tool_calls = 1
        if index == 1:
            path = workspace / f"facts_{index:02d}.txt"
            lines = [f"line1=ignore-{index:02d}", f"line2=ignore-{index:02d}", f"owner=user-{index:02d}"]
            _write(path, "\n".join(lines) + "\n")
            answer = lines[2]
            prompt = f"Read {path.name} and return exactly the full text on line 3. No extra words."
            read_paths = [path]
        elif index == 2:
            path = workspace / "profile.txt"
            _write(path, "name=alice\nteam=blue\n")
            answer = '{"name":"alice","team":"blue","city":""}'
            prompt = f"Read {path.name}. Return exactly this JSON shape with no extra fields: {{\"name\":\"...\",\"team\":\"...\",\"city\":\"...\"}}. Use an empty string when a field is missing."
            read_paths = [path]
        elif index == 3:
            left = workspace / "left.txt"
            right = workspace / "right.txt"
            _write(left, "status=ready\n")
            _write(right, "status=blocked\n")
            answer = "status=conflict"
            prompt = f"Read {left.name} and {right.name}. If the status values disagree, reply exactly status=conflict."
            read_paths = [left, right]
            min_tool_calls = 2
        elif index == 4:
            path = workspace / "profile_guard.txt"
            _write(path, "name=carol\n")
            answer = '{"name":"carol","city":"","unsupported":["city"]}'
            prompt = (
                f"Read {path.name}. Return exact JSON with keys name, city, unsupported. "
                "Do not invent missing fields. If city is missing, use an empty string and list city under unsupported."
            )
            read_paths = [path]
        else:
            a = workspace / "facts_a.txt"
            b = workspace / "facts_b.txt"
            _write(a, "alpha=1\nbeta=2\n")
            _write(b, "alpha=1\nbeta=3\n")
            answer = '{"alpha":"1","beta_conflict":true}'
            prompt = f"Read {a.name} and {b.name}. Return exact JSON {{\"alpha\":\"1\",\"beta_conflict\":true}} if alpha matches and beta disagrees."
            read_paths = [a, b]
            min_tool_calls = 2

        contract = BenchmarkVerificationContract(
            task_type="reading",
            expected_answer=None if index in {2, 4, 5} else answer,
            expected_json=(
                {"name": "alice", "team": "blue", "city": ""}
                if index == 2
                else (
                    {"name": "carol", "city": "", "unsupported": ["city"]}
                    if index == 4
                    else ({"alpha": "1", "beta_conflict": True} if index == 5 else None)
                )
            ),
            required_history_events=["tool_called", "tool_result", "verification_passed"],
            required_tools_used=required_tools,
            min_tool_calls=min_tool_calls,
            forbid_unexpected_workspace_changes=True,
        )
        if not scripted:
            return TaskScenario(prompt=prompt, workspace=workspace, model_client=None, verification_contract=contract)
        plan = _plan_response(
            goal=prompt,
            steps=[
                *[
                    _plan_step(
                        f"step_read_{read_index}",
                        f"Read file {read_index}",
                        "read",
                        expected_tool="read_text",
                        expected_output="facts",
                        success_criteria="facts read",
                        depends_on=[] if read_index == 1 else [f"step_read_{read_index - 1}"],
                    )
                    for read_index in range(1, len(read_paths) + 1)
                ],
                _plan_step(
                    "step_answer",
                    "Answer",
                    "respond",
                    expected_output=answer,
                    success_criteria="reply exact answer",
                    depends_on=[f"step_read_{len(read_paths)}"],
                ),
            ],
        )
        responses = [plan]
        responses.extend(_tool_call("read_text", {"path": str(path)}) for path in read_paths)
        responses.append(answer)
        client = ScriptedBenchmarkClient(responses=responses)
        return TaskScenario(prompt=prompt, workspace=workspace, model_client=client, verification_contract=contract)

    return BenchmarkTaskDefinition(
        task_id=task_id,
        task_type="reading",
        description="Live benchmark exact file-reading extraction.",
        build=lambda workspace: _scenario(workspace, scripted=True),
        build_live=lambda workspace: _scenario(workspace, scripted=False),
        difficulty="easy" if index == 1 else ("medium" if index in {2, 3} else "hard"),
        tags=[
            "live-subset",
            "reading",
            "environment",
            "filesystem",
            *(["verification-edge"] if index in {4, 5} else []),
            *(["false-positive-killer"] if index in {4, 5} else []),
        ],
        setup_instructions=[
            "Create a short file-reading task with mechanically checkable extraction.",
            *(["Use more than one file for harder contradiction and guard tasks."] if index in {3, 5} else []),
        ],
        config_overrides={},
    )


def _live_multi_step_definition(index: int) -> BenchmarkTaskDefinition:
    task_id = f"live_multi_step_{index:02d}"
    answer = f"written-live-{index:02d}"

    def _scenario(workspace: Path, *, scripted: bool) -> TaskScenario:
        numbers = workspace / f"numbers_{index:02d}.txt"
        result_file = workspace / f"result_{index:02d}.txt"
        left = 20 + index
        right = 22 + index
        total = left + right
        _write(numbers, f"{left}\n{right}\n")
        required_tools = ["read_text", "calculator", "write_file"]
        min_tool_calls = 3
        if index in {1, 2}:
            prompt = (
                f"Read {numbers.name}. Use the calculator tool to add the two numbers on lines 1 and 2. "
                f"Create {result_file.name} containing exactly sum={total} followed by a newline. Reply exactly {answer}."
            )
            extra_steps: list[dict[str, object]] = []
            scripted_calls: list[object] = []
        elif index == 3:
            prompt = (
                f"Read {numbers.name}. Use the calculator tool to add the two numbers. "
                f"Create {result_file.name} containing exactly sum={total} followed by a newline, then reread {result_file.name} before replying exactly {answer}."
            )
            extra_steps = [
                _plan_step("step_verify", "Reread result", "read", expected_tool="read_text", expected_output="written result", success_criteria="result reread", depends_on=["step_write"]),
            ]
            scripted_calls = [_tool_call("read_text", {"path": str(result_file)})]
            min_tool_calls = 4
        elif index == 4:
            audit = workspace / "audit.txt"
            _write(audit, "status=pending\n")
            prompt = (
                f"Read {numbers.name}. Use the calculator tool to add the two numbers. "
                f"Create {result_file.name} containing exactly sum={total} followed by a newline. "
                f"Then update {audit.name} so it becomes exactly status=verified followed by a newline. Reply exactly {answer}."
            )
            extra_steps = [
                _plan_step("step_audit", "Update audit", "write", expected_tool="write_file", expected_output="audit file", success_criteria="audit updated", depends_on=["step_write"]),
            ]
            scripted_calls = [_tool_call("write_file", {"path": str(audit), "content": "status=verified\n"})]
            min_tool_calls = 4
        else:
            expected_files = workspace / "expected_sum.txt"
            _write(expected_files, f"sum={total}\n")
            test_file = workspace / "test_result_match.py"
            prompt = (
                f"Read {numbers.name}. Use the calculator tool to add the two numbers. "
                f"Create {result_file.name} containing exactly sum={total} followed by a newline. "
                f"Then run tests in {test_file.name} that compare {result_file.name} with {expected_files.name}. Reply exactly {answer}."
            )
            required_tools = ["read_text", "calculator", "write_file", "run_tests"]
            extra_steps = [
                _plan_step("step_tests", "Run verification test", "tool", expected_tool="run_tests", expected_output="passing verification", success_criteria="verification passes", depends_on=["step_write"]),
            ]
            _write(
                test_file,
                (
                    "import unittest\n\n"
                    "from pathlib import Path\n\n\n"
                    "class ResultMatchTests(unittest.TestCase):\n"
                    "    def test_result_matches_expected(self) -> None:\n"
                    f"        self.assertEqual(Path({result_file.name!r}).read_text(encoding='utf-8'), Path({expected_files.name!r}).read_text(encoding='utf-8'))\n\n\n"
                    "if __name__ == '__main__':\n"
                    "    unittest.main()\n"
                ),
            )
            scripted_calls = [_tool_call("run_tests", {"command": ["python3", "-m", "unittest", "-q", str(test_file.name)]})]
            min_tool_calls = 4

        expected_files = {result_file: f"sum={total}\n"}
        if index == 4:
            expected_files[audit] = "status=verified\n"
        contract = BenchmarkVerificationContract(
            task_type="multi_step",
            expected_answer=answer,
            expected_files=expected_files,
            required_history_events=["tool_called", "tool_result", "verification_passed"],
            required_tools_used=required_tools,
            min_tool_calls=min_tool_calls,
            forbid_unexpected_workspace_changes=True,
        )
        if not scripted:
            return TaskScenario(prompt=prompt, workspace=workspace, model_client=None, verification_contract=contract)
        plan = _plan_response(
            goal=prompt,
            steps=[
                _plan_step("step_read", "Read numbers", "read", expected_tool="read_text", expected_output="numbers", success_criteria="numbers read"),
                _plan_step("step_calc", "Compute sum", "tool", expected_tool="calculator", expected_output="sum", success_criteria="sum computed", depends_on=["step_read"]),
                _plan_step("step_write", "Write result", "write", expected_tool="write_file", expected_output="result file", success_criteria="result file written", depends_on=["step_calc"]),
                *extra_steps,
                _plan_step(
                    "step_answer",
                    "Answer",
                    "respond",
                    expected_output=answer,
                    success_criteria="reply exact answer",
                    depends_on=[str(extra_steps[-1]["step_id"]) if extra_steps else "step_write"],
                ),
            ],
        )
        responses = [
            plan,
            _tool_call("read_text", {"path": str(numbers)}),
            _tool_call("calculator", {"expression": f"{left} + {right}"}),
            _tool_call("write_file", {"path": str(result_file), "content": f"sum={total}\n", "create": True}),
            *scripted_calls,
            answer,
        ]
        client = ScriptedBenchmarkClient(responses=responses)
        return TaskScenario(prompt=prompt, workspace=workspace, model_client=client, verification_contract=contract)

    return BenchmarkTaskDefinition(
        task_id=task_id,
        task_type="multi_step",
        description="Live benchmark read-compute-write task with deterministic file verification.",
        build=lambda workspace: _scenario(workspace, scripted=True),
        build_live=lambda workspace: _scenario(workspace, scripted=False),
        difficulty="medium" if index in {1, 2} else "hard",
        tags=[
            "live-subset",
            "multi-step",
            "environment",
            "calculator",
            *(["long-run", "recovery"] if index in {3, 4, 5} else []),
            *(["verification-edge"] if index == 5 else []),
        ],
        setup_instructions=[
            "Create a multi-step read-compute-write task with deterministic output verification.",
            *(["Require at least one extra verification or correction step for the harder live tasks."] if index in {3, 4, 5} else []),
        ],
        config_overrides={"tools_allow_side_effect_tools": True, "runtime_max_tool_steps": 8, "runtime_tool_call_budget": 8},
    )


def _live_failure_definition(index: int) -> BenchmarkTaskDefinition:
    task_id = f"live_failure_{index:02d}"

    def _scenario(workspace: Path, *, scripted: bool) -> TaskScenario:
        answer = "not done"
        if index == 1:
            missing = workspace / "missing.txt"
            prompt = f"Read {missing.name}. If the file does not exist or you cannot verify its contents, reply exactly not done. Do not guess."
            oracle = None
        elif index == 2:
            facts = workspace / "facts.txt"
            _write(facts, "owner=carol\n")
            prompt = f"Read {facts.name}. Only return owner=bob if the file proves it. If you cannot verify that, reply exactly not done."
            oracle = None
        elif index == 3:
            broken = workspace / "broken.py"
            test_name = "test_broken.py"
            _write(broken, "def value() -> int:\n    return 0\n")
            _write(workspace / test_name, "import unittest\nfrom broken import value\n\nclass BrokenTests(unittest.TestCase):\n    def test_value(self) -> None:\n        self.assertEqual(value(), 5)\n\nif __name__ == '__main__':\n    unittest.main()\n")
            prompt = f"Run tests in {test_name}. If they fail or you cannot verify success, reply exactly not done."
            oracle = None
        elif index == 4:
            a = workspace / "left.txt"
            b = workspace / "right.txt"
            _write(a, "answer=5\n")
            _write(b, "answer=9\n")
            prompt = f"Read {a.name} and {b.name}. If the files disagree, reply exactly not done."
            oracle = None
        else:
            prompt = "Read the requested file and return line 2. If no file path is given or you cannot verify the answer, reply exactly not done."
            oracle = None
        contract = BenchmarkVerificationContract(
            task_type="failure",
            expected_answer=answer,
            required_history_events=["reasoning_completed"],
        )
        if not scripted:
            return TaskScenario(prompt=prompt, workspace=workspace, model_client=None, verification_contract=contract, oracle=oracle)
        client = ScriptedBenchmarkClient(
            contract_responses={
                "plan": [_plan_response(goal=prompt, steps=[_plan_step("step_answer", "Answer", "respond", expected_output=answer, success_criteria="reply not done")])],
                "plain_text": [answer],
            }
        )
        return TaskScenario(prompt=prompt, workspace=workspace, model_client=client, verification_contract=contract, oracle=oracle)

    return BenchmarkTaskDefinition(
        task_id=task_id,
        task_type="failure",
        description="Live benchmark safe failure / not-done task.",
        build=lambda workspace: _scenario(workspace, scripted=True),
        build_live=lambda workspace: _scenario(workspace, scripted=False),
        difficulty="easy" if index in {1, 2} else ("medium" if index == 3 else "hard"),
        tags=[
            "live-subset",
            "failure",
            "not-done",
            "adversarial",
            *(["false-positive-killer"] if index in {2, 4, 5} else []),
            *(["verification-edge"] if index in {3, 4} else []),
            *(["long-run", "recovery"] if index in {3, 4, 5} else []),
        ],
        setup_instructions=["Prepare a task that must fail safely or return not done instead of guessing."],
        config_overrides={"tools_allow_side_effect_tools": True},
    )


def _live_quality_definition(index: int) -> BenchmarkTaskDefinition:
    task_id = f"live_quality_{index:02d}"

    def _scenario(workspace: Path, *, scripted: bool) -> TaskScenario:
        if index == 1:
            debug_log = workspace / "bench-debug.log"
            _write(debug_log, "ERROR code=104\nWARNING slow=true\n")
            prompt = f"Read {debug_log.name}. Reply exactly ERROR code=104. No extra words."
            answer = "ERROR code=104"
            oracle = PromptUnderstandingOracle(task_type="structured", completeness="complete", expand_task=False, split_task=True, strategy_profile="reading", detected_entities_contains=["bench-debug.log"], detected_goals_contains=["read"])
        elif index == 2:
            facts = workspace / "facts.txt"
            _write(facts, "alpha=1\nbeta=2\n")
            prompt = f"1. Read {facts.name}\n2. Reply exactly beta=2\nDo not add anything else."
            answer = "beta=2"
            oracle = PromptUnderstandingOracle(task_type="already_decomposed", completeness="complete", requires_expansion=False, requires_decomposition=True, expand_task=False, split_task=True, ask_user=False, strategy_profile="reading", detected_entities_contains=["facts.txt"])
        elif index == 3:
            prompt = "Read the requested file and return line 2. If no file path is given, reply exactly not done."
            answer = "not done"
            oracle = PromptUnderstandingOracle(
                task_type="incomplete",
                completeness="incomplete",
                requires_decomposition=False,
                split_task=False,
                strategy_profile="reading",
                detected_goals_contains=["read"],
            )
        elif index == 4:
            values = workspace / "values.txt"
            _write(values, "18\n24\n")
            prompt = f"1. Read {values.name}\n2. Use the calculator tool to add the numbers on lines 1 and 2\n3. Reply with only the number."
            answer = "42"
            oracle = PromptUnderstandingOracle(task_type="already_decomposed", completeness="complete", requires_expansion=False, requires_decomposition=True, expand_task=False, split_task=True, ask_user=False, strategy_profile="reading", detected_entities_contains=["values.txt"], detected_goals_contains=["read"])
        else:
            status = workspace / "ready.txt"
            _write(status, "flag=1\n")
            prompt = f"1. Read {status.name}\n2. Reply exactly flag=1\nDo not add anything else."
            answer = "flag=1"
            oracle = PromptUnderstandingOracle(task_type="already_decomposed", completeness="complete", requires_expansion=False, requires_decomposition=True, expand_task=False, split_task=True, ask_user=False, strategy_profile="reading", detected_entities_contains=["ready.txt"], detected_goals_contains=["read"])
        contract = BenchmarkVerificationContract(
            task_type="quality",
            expected_answer=answer,
            required_history_events=["prompt_analyzed", "decision_made", "reasoning_completed"],
        )
        if not scripted:
            return TaskScenario(prompt=prompt, workspace=workspace, model_client=None, verification_contract=contract, oracle=oracle)
        if index == 4:
            client = ScriptedBenchmarkClient(
                responses=[
                    _plan_response(goal=prompt, steps=[_plan_step("step_read", "Read values", "read", expected_tool="read_text", expected_output="values", success_criteria="values read"), _plan_step("step_calc", "Compute sum", "tool", expected_tool="calculator", expected_output="sum", success_criteria="sum computed", depends_on=["step_read"]), _plan_step("step_answer", "Answer", "respond", expected_output=answer, success_criteria="reply exact number", depends_on=["step_calc"])]),
                    _tool_call("read_text", {"path": str(values)}),
                    _tool_call("calculator", {"expression": "18 + 24"}),
                    answer,
                ]
            )
        elif index in {1, 2, 5}:
            target = debug_log if index == 1 else facts if index == 2 else status
            client = ScriptedBenchmarkClient(
                responses=[
                    _plan_response(goal=prompt, steps=[_plan_step("step_read", "Read file", "read", expected_tool="read_text", expected_output="file", success_criteria="file read"), _plan_step("step_answer", "Answer", "respond", expected_output=answer, success_criteria="reply exact answer", depends_on=["step_read"])]),
                    _tool_call("read_text", {"path": str(target)}),
                    answer,
                ]
            )
        else:
            client = ScriptedBenchmarkClient(responses=[_plan_response(goal=prompt, steps=[_plan_step("step_answer", "Answer", "respond", expected_output=answer, success_criteria="reply exact answer")]), answer])
        return TaskScenario(prompt=prompt, workspace=workspace, model_client=client, verification_contract=contract, oracle=oracle)

    return BenchmarkTaskDefinition(
        task_id=task_id,
        task_type="quality",
        description="Live benchmark prompt-understanding / quality task.",
        build=lambda workspace: _scenario(workspace, scripted=True),
        build_live=lambda workspace: _scenario(workspace, scripted=False),
        difficulty="easy" if index == 1 else ("medium" if index in {2, 4, 5} else "hard"),
        tags=[
            "live-subset",
            "quality",
            "prompt-understanding",
            *(["ambiguity"] if index in {2, 3, 5} else []),
            *(["false-positive-killer", "verification-edge"] if index in {3, 4} else []),
        ],
        setup_instructions=["Prepare a prompt-understanding task that must not be over-expanded or accepted without sufficient evidence."],
        config_overrides={"tools_allow_side_effect_tools": True},
    )


def generated_live_subset_tasks() -> list[BenchmarkTaskDefinition]:
    tasks: list[BenchmarkTaskDefinition] = []
    tasks.extend(_live_coding_definition(index) for index in range(1, 6))
    tasks.extend(_live_file_edit_definition(index) for index in range(1, 6))
    tasks.extend(_live_reading_definition(index) for index in range(1, 6))
    tasks.extend(_live_multi_step_definition(index) for index in range(1, 6))
    tasks.extend(_live_failure_definition(index) for index in range(1, 6))
    tasks.extend(_live_quality_definition(index) for index in range(1, 6))
    validate_live_subset_catalog(tasks)
    return tasks


# The live subset is a fixed-size representative slice of the full benchmark.
# These quotas prevent silent drift toward only the easiest live-capable tasks.
LIVE_SUBSET_TASK_TYPE_MINIMUMS: dict[str, int] = {
    "coding": 5,
    "file_edit": 5,
    "reading": 5,
    "multi_step": 5,
    "failure": 5,
    "quality": 5,
}

LIVE_SUBSET_DIFFICULTY_MINIMUMS: dict[str, int] = {
    "easy": 6,
    "medium": 12,
    "hard": 12,
}

LIVE_SUBSET_STRUCTURAL_MINIMUMS: dict[str, int] = {
    "multifile_coding": 3,
    "long_run": 3,
    "false_positive_killer": 3,
    "verification_edge": 3,
    "prompt_understanding": 3,
    "environment_or_shell": 3,
}


def validate_live_subset_catalog(tasks: list[BenchmarkTaskDefinition]) -> None:
    type_counts: dict[str, int] = {}
    difficulty_counts: dict[str, int] = {}
    for task in tasks:
        type_counts[task.task_type] = type_counts.get(task.task_type, 0) + 1
        difficulty_counts[task.difficulty] = difficulty_counts.get(task.difficulty, 0) + 1
    if len(tasks) < 30:
        raise ValueError(f"Live benchmark subset must contain at least 30 tasks, found {len(tasks)}")
    missing_types = {
        task_type: minimum
        for task_type, minimum in LIVE_SUBSET_TASK_TYPE_MINIMUMS.items()
        if type_counts.get(task_type, 0) < minimum
    }
    if missing_types:
        raise ValueError(f"Live benchmark subset does not meet category minimums: {missing_types}")
    missing_difficulties = {
        difficulty: minimum
        for difficulty, minimum in LIVE_SUBSET_DIFFICULTY_MINIMUMS.items()
        if difficulty_counts.get(difficulty, 0) < minimum
    }
    if missing_difficulties:
        raise ValueError(f"Live benchmark subset does not meet difficulty minimums: {missing_difficulties}")
    structural_counts = {
        "multifile_coding": sum(1 for task in tasks if task.task_type == "coding" and "multifile" in task.tags),
        "long_run": sum(1 for task in tasks if "long-run" in task.tags),
        "false_positive_killer": sum(1 for task in tasks if "false-positive-killer" in task.tags),
        "verification_edge": sum(1 for task in tasks if "verification-edge" in task.tags),
        "prompt_understanding": sum(1 for task in tasks if "prompt-understanding" in task.tags or "ambiguity" in task.tags),
        "environment_or_shell": sum(1 for task in tasks if {"environment", "shell", "filesystem"} & set(task.tags)),
    }
    missing_structural = {
        key: minimum
        for key, minimum in LIVE_SUBSET_STRUCTURAL_MINIMUMS.items()
        if structural_counts.get(key, 0) < minimum
    }
    if missing_structural:
        raise ValueError(f"Live benchmark subset does not meet representativeness minimums: {missing_structural}")
    ids = [task.task_id for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("Live benchmark subset contains duplicate task ids")
