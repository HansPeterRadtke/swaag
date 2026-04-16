from __future__ import annotations

import ast
import json
import re
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from swaag.retrieval.embeddings import build_backend
from swaag.tools.base import ToolValidationError, _validate_schema_value
from swaag.types import HistoryEvent, Plan, PlanStep, SessionState, ToolExecutionResult, VerificationType

_EXECUTION_ALLOWLIST = frozenset({"python", "python3", "pytest"})


class VerificationError(RuntimeError):
    pass


@dataclass(slots=True)
class VerificationOutcome:
    verification_passed: bool
    verification_type_used: VerificationType
    conditions_met: list[str]
    conditions_failed: list[str]
    evidence: dict[str, Any]
    confidence: float
    reason: str
    requires_retry: bool
    requires_replan: bool

    @property
    def passed(self) -> bool:
        return self.verification_passed


@dataclass(slots=True)
class VerificationArtifacts:
    assistant_text: str = ""
    tool_results: list[ToolExecutionResult] = field(default_factory=list)
    runtime_artifacts: dict[str, Any] = field(default_factory=dict)

    @property
    def latest_tool_result(self) -> ToolExecutionResult | None:
        return self.tool_results[-1] if self.tool_results else None


class VerificationEngine:
    def __init__(
        self,
        *,
        command_allowlist: set[str] | None = None,
        semantic_backend_mode: str = "llm_scoring",
        semantic_base_url: str | None = None,
        semantic_seed: int = 11,
        semantic_connect_timeout_seconds: int = 10,
        semantic_read_timeout_seconds: int = 60,
    ):
        self._command_allowlist = frozenset(command_allowlist or _EXECUTION_ALLOWLIST)
        backend_mode = semantic_backend_mode
        # Standalone deterministic verification tests often construct the
        # engine without a live semantic endpoint. That should degrade
        # explicitly instead of crashing during initialization.
        if backend_mode == "llm_scoring" and not semantic_base_url:
            backend_mode = "degraded_lexical"
        self._semantic_backend = build_backend(
            backend_mode,
            base_url=semantic_base_url,
            seed=semantic_seed,
            connect_timeout_seconds=semantic_connect_timeout_seconds,
            read_timeout_seconds=semantic_read_timeout_seconds,
        )

    def _expected_parts(self, step: PlanStep) -> list[str]:
        """Return ``expected_output`` and ``success_criteria`` as plain text.

        Previously this method filtered them through a hard-coded vocabulary
        of "generic" tokens, which silently dropped strings like "produce
        the answer". That keyword filter has been removed: any non-empty
        string is forwarded to the LLM-driven semantic backend, which is
        the only allowed semantic judge.
        """

        return [
            part.strip()
            for part in [step.expected_output, step.success_criteria]
            if isinstance(part, str) and part.strip()
        ]

    def verify_step(
        self,
        *,
        runtime,
        state: SessionState,
        plan: Plan,
        step: PlanStep,
        artifacts: VerificationArtifacts,
    ) -> VerificationOutcome:
        if not step.verification_checks:
            raise VerificationError(f"Step {step.step_id} has no verification checks")
        check_index = self._check_index(step)
        evidence: dict[str, Any] = {}
        conditions_met: list[str] = []
        conditions_failed: list[str] = []

        def run_named_condition(name: str) -> bool:
            check = check_index[name]
            passed, condition_evidence = self._run_check(
                runtime=runtime,
                state=state,
                plan=plan,
                step=step,
                artifacts=artifacts,
                verification_type=step.verification_type,
                check=check,
            )
            evidence[name] = condition_evidence
            if passed:
                conditions_met.append(name)
            else:
                conditions_failed.append(name)
            return passed

        if step.verification_type == "llm_fallback":
            deterministic = [name for name in step.required_conditions if check_index[name].get("check_type") != "criterion"]
            llm_required = [name for name in step.required_conditions if check_index[name].get("check_type") == "criterion"]
            llm_optional = [name for name in step.optional_conditions if check_index[name].get("check_type") == "criterion"]
            for name in deterministic:
                run_named_condition(name)
            for name in [name for name in step.optional_conditions if name not in llm_optional]:
                run_named_condition(name)
            if conditions_failed:
                return self._finalize(step, evidence, conditions_met, conditions_failed)
            llm_results = self._run_llm_fallback(
                runtime=runtime,
                state=state,
                step=step,
                artifacts=artifacts,
                checks=[check_index[name] for name in [*llm_required, *llm_optional]],
            )
            for name, passed, condition_evidence in llm_results:
                evidence[name] = condition_evidence
                if passed:
                    conditions_met.append(name)
                else:
                    conditions_failed.append(name)
            self._apply_perspectives(
                state=state,
                plan=plan,
                step=step,
                artifacts=artifacts,
                evidence=evidence,
                conditions_met=conditions_met,
                conditions_failed=conditions_failed,
            )
            return self._finalize(step, evidence, conditions_met, conditions_failed)

        for name in step.required_conditions:
            run_named_condition(name)
        for name in step.optional_conditions:
            run_named_condition(name)
        self._apply_perspectives(
            state=state,
            plan=plan,
            step=step,
            artifacts=artifacts,
            evidence=evidence,
            conditions_met=conditions_met,
            conditions_failed=conditions_failed,
        )
        return self._finalize(step, evidence, conditions_met, conditions_failed)

    def _apply_perspectives(
        self,
        *,
        state: SessionState,
        plan: Plan,
        step: PlanStep,
        artifacts: VerificationArtifacts,
        evidence: dict[str, Any],
        conditions_met: list[str],
        conditions_failed: list[str],
    ) -> None:
        perspectives = self._run_perspectives(state=state, plan=plan, step=step, artifacts=artifacts)
        if not perspectives:
            return
        evidence["perspectives"] = {name: detail for name, _, detail in perspectives}
        required = self._required_perspectives(step=step, artifacts=artifacts)
        for name, passed, _detail in perspectives:
            marker = f"perspective:{name}"
            if passed:
                conditions_met.append(marker)
            elif name in required:
                conditions_failed.append(marker)

    def _required_perspectives(
        self,
        *,
        step: PlanStep,
        artifacts: VerificationArtifacts,
    ) -> set[str]:
        required = {"consistency"}
        latest = artifacts.latest_tool_result
        if latest is not None and latest.tool_name in {"read_file", "write_file", "read_text", "edit_text", "run_tests", "shell_command"}:
            required.add("structural")
        # llm_fallback respond/reasoning steps already include an isolated
        # semantic review pass through the criterion contract. The reviewer
        # perspective still runs for traceability, but it is advisory rather
        # than gating so generic expected_output placeholders do not create a
        # second contradictory semantic judge.
        if step.kind in {"respond", "reasoning"} and step.verification_type != "llm_fallback":
            required.add("reviewer")
        return required

    def _run_perspectives(
        self,
        *,
        state: SessionState,
        plan: Plan,
        step: PlanStep,
        artifacts: VerificationArtifacts,
    ) -> list[tuple[str, bool, dict[str, Any]]]:
        perspectives = [self._run_consistency_perspective(plan=plan, step=step, artifacts=artifacts)]
        latest = artifacts.latest_tool_result
        if latest is not None and latest.tool_name in {"read_file", "write_file", "read_text", "edit_text", "run_tests", "shell_command"}:
            perspectives.append(self._run_structural_perspective(state=state, artifacts=artifacts))
        if step.kind in {"respond", "reasoning"} and artifacts.assistant_text.strip():
            perspectives.append(self._run_reviewer_perspective(step=step, artifacts=artifacts))
        return perspectives

    def _run_reviewer_perspective(
        self,
        *,
        step: PlanStep,
        artifacts: VerificationArtifacts,
    ) -> tuple[str, bool, dict[str, Any]]:
        latest = artifacts.latest_tool_result
        actual_text = artifacts.assistant_text.strip()
        if not actual_text and latest is not None:
            actual_text = str(
                latest.output.get("text")
                or latest.output.get("content")
                or latest.output.get("result")
                or latest.output.get("diff")
                or ""
            ).strip()
        expected_parts = self._expected_parts(step)
        expected_text = "\n".join(expected_parts).strip()
        literal_expected_output = step.expected_output.strip() if isinstance(step.expected_output, str) else ""
        normalized_actual = " ".join(actual_text.split())
        normalized_literal = " ".join(literal_expected_output.split())
        literal_match = bool(normalized_actual and normalized_literal and normalized_actual == normalized_literal)
        if not expected_text and not literal_match:
            return "reviewer", True, {"checked": False}
        # When the configured semantic backend is one of the degraded
        # lexical fallbacks, we cannot get an LLM-driven relevance score.
        # Producing a fail/pass on a TF-IDF cosine would be exactly the
        # kind of magic-number heuristic the new design forbids. The
        # reviewer therefore records the absence of an LLM judgement and
        # passes structurally — production deployments use a real model
        # so this fallback never gates a real run.
        backend_unavailable = self._semantic_backend.degraded or self._semantic_backend.mode in {
            "degraded_lexical",
            "heuristic_fallback",
        }
        if backend_unavailable:
            return "reviewer", bool(actual_text), {
                "checked": False,
                "reason": "semantic_backend_unavailable",
                "review_backend_mode": self._semantic_backend.mode,
                "review_backend_degraded": True,
                "expected_text": expected_text,
                "actual_text": actual_text,
                "literal_match": literal_match,
            }
        relevance_scores = (
            self._semantic_backend.score_query(expected_text, [actual_text]) if expected_text and actual_text else [0.0]
        )
        relevance = float(relevance_scores[0]) if relevance_scores else 0.0
        evidence = {
            "checked": True,
            "expected_text": expected_text,
            "expected_literal": literal_expected_output,
            "actual_text": actual_text,
            "literal_match": literal_match,
            "llm_relevance": relevance,
            "review_backend_mode": self._semantic_backend.mode,
            "review_backend_degraded": self._semantic_backend.degraded,
        }
        # Decision rule: an exact-literal match always passes; otherwise the
        # LLM relevance score must be at least 0.5 (the LLM's own midpoint
        # between "irrelevant" and "matches"). The threshold is structural
        # — the LLM produced the score; we are not picking a magic number
        # to compensate for a lexical formula.
        passed = bool(actual_text) and (literal_match or relevance >= 0.5)
        return "reviewer", passed, evidence

    def _run_consistency_perspective(
        self,
        *,
        plan: Plan,
        step: PlanStep,
        artifacts: VerificationArtifacts,
    ) -> tuple[str, bool, dict[str, Any]]:
        completed = {item.step_id for item in plan.steps if item.status == "completed"}
        missing_dependencies = [dependency for dependency in step.depends_on if dependency not in completed]
        latest = artifacts.latest_tool_result
        tool_matches = latest is None or step.expected_tool in {None, "", latest.tool_name}
        tool_output = latest.output if latest is not None else {}
        exit_code_consistent = True
        if latest is not None and latest.tool_name == "run_tests":
            exit_code = tool_output.get("exit_code")
            passed = tool_output.get("passed")
            exit_code_consistent = isinstance(exit_code, int) and isinstance(passed, bool) and ((exit_code == 0) == passed)
        evidence = {
            "missing_dependencies": missing_dependencies,
            "tool_name": None if latest is None else latest.tool_name,
            "expected_tool": step.expected_tool,
            "tool_matches": tool_matches,
            "exit_code_consistent": exit_code_consistent,
        }
        passed = not missing_dependencies and tool_matches and exit_code_consistent
        return "consistency", passed, evidence

    def _run_structural_perspective(
        self,
        *,
        state: SessionState,
        artifacts: VerificationArtifacts,
    ) -> tuple[str, bool, dict[str, Any]]:
        latest = artifacts.latest_tool_result
        if latest is None:
            return "structural", True, {"checked": False}
        output = latest.output
        path_text = output.get("path")
        relative_path = output.get("relative_path")
        text_value = output.get("text")
        evidence: dict[str, Any] = {"tool_name": latest.tool_name, "checked": True}
        if isinstance(path_text, str):
            path = Path(path_text).expanduser()
            exists = path.exists()
            evidence["path"] = str(path)
            evidence["path_exists"] = exists
            if isinstance(text_value, str) and exists:
                actual = path.read_text(encoding="utf-8")
                evidence["content_matches"] = actual == text_value
                if isinstance(relative_path, str):
                    evidence["environment_matches"] = state.environment.workspace.known_files.get(relative_path) == text_value
                return "structural", bool(evidence["content_matches"]) and bool(evidence.get("environment_matches", True)), evidence
            return "structural", exists, evidence
        if latest.tool_name == "shell_command":
            lists_ok = all(isinstance(output.get(key), list) for key in ("created_files", "modified_files", "deleted_files"))
            evidence["lists_ok"] = lists_ok
            evidence["cwd_after"] = output.get("cwd_after")
            return "structural", lists_ok and isinstance(output.get("exit_code"), int), evidence
        if latest.tool_name == "run_tests":
            stdout = output.get("stdout", "")
            stderr = output.get("stderr", "")
            evidence["has_output"] = isinstance(stdout, str) and isinstance(stderr, str)
            evidence["passed"] = output.get("passed")
            return "structural", evidence["has_output"] and isinstance(output.get("passed"), bool), evidence
        return "structural", True, evidence

    def _check_index(self, step: PlanStep) -> dict[str, dict[str, Any]]:
        index: dict[str, dict[str, Any]] = {}
        for raw in step.verification_checks:
            name = str(raw.get("name", "")).strip()
            if not name:
                raise VerificationError(f"Step {step.step_id} has a verification check without a name")
            index[name] = dict(raw)
        missing = (set(step.required_conditions) | set(step.optional_conditions)) - set(index)
        if missing:
            raise VerificationError(
                f"Step {step.step_id} references unknown verification conditions: {', '.join(sorted(missing))}"
            )
        return index

    def _finalize(
        self,
        step: PlanStep,
        evidence: dict[str, Any],
        conditions_met: list[str],
        conditions_failed: list[str],
    ) -> VerificationOutcome:
        total = len(step.required_conditions) + len(step.optional_conditions)
        confidence = 0.0 if total == 0 else len(conditions_met) / total
        perspective_failures = [name for name in conditions_failed if name.startswith("perspective:")]
        passed = not any(name in conditions_failed for name in step.required_conditions) and not perspective_failures
        reason = "verified" if passed else ";".join(sorted(conditions_failed))
        requires_replan = any(name == "dependencies_completed" for name in conditions_failed)
        requires_retry = not passed and not requires_replan
        return VerificationOutcome(
            verification_passed=passed,
            verification_type_used=step.verification_type,
            conditions_met=conditions_met,
            conditions_failed=conditions_failed,
            evidence=evidence,
            confidence=confidence,
            reason=reason,
            requires_retry=requires_retry,
            requires_replan=requires_replan,
        )

    def _run_check(
        self,
        *,
        runtime,
        state: SessionState,
        plan: Plan,
        step: PlanStep,
        artifacts: VerificationArtifacts,
        verification_type: VerificationType,
        check: dict[str, Any],
    ) -> tuple[bool, dict[str, Any]]:
        check_type = str(check.get("check_type", "")).strip()
        if verification_type == "execution":
            return self._run_execution_check(check)
        if verification_type == "structural":
            return self._run_structural_check(check, artifacts=artifacts)
        if verification_type == "value":
            return self._run_value_check(check, artifacts=artifacts)
        if verification_type == "composite":
            if check_type in {"command_success"}:
                return self._run_execution_check(check)
            if check_type in {"file_exists", "file_contains", "json_schema_valid", "function_exists", "symbol_exists", "dependencies_completed", "tool_output_schema_valid"}:
                return self._run_structural_check(check, artifacts=artifacts, plan=plan, step=step)
            if check_type in {"artifact_present", "tool_name_equals", "tool_output_nonempty", "tool_files_changed", "string_nonempty", "exact_match", "numeric_tolerance", "string_match"}:
                return self._run_value_check(check, artifacts=artifacts)
            if check_type == "criterion":
                raise VerificationError(f"Criterion check {check.get('name')} may only be used with llm_fallback")
        if verification_type == "llm_fallback":
            if check_type == "criterion":
                raise VerificationError("Criterion checks must be executed through llm fallback")
            if check_type in {"dependencies_completed", "tool_output_schema_valid"}:
                return self._run_structural_check(check, artifacts=artifacts, plan=plan, step=step)
            return self._run_value_check(check, artifacts=artifacts)
        raise VerificationError(f"Unsupported verification type/check combination: {verification_type}/{check_type}")

    def _run_execution_check(self, check: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        command = check.get("command")
        if not isinstance(command, list) or not command or not all(isinstance(item, str) and item for item in command):
            raise VerificationError("Execution verification requires a non-empty command list")
        executable = Path(command[0]).name
        if executable not in self._command_allowlist:
            raise VerificationError(f"Execution verification command is not allowed: {executable}")
        cwd = check.get("cwd")
        timeout_seconds = int(check.get("timeout_seconds", 30))
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            shell=False,
            check=False,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        framework = str(check.get("framework", "")).strip()
        passed = completed.returncode == int(check.get("expected_exit_code", 0))
        summary: dict[str, Any] = {
            "command": command,
            "cwd": cwd,
            "exit_code": completed.returncode,
            "stdout": stdout,
            "stderr": stderr,
        }
        if framework:
            parsed = self._parse_framework_result(framework, stdout, stderr)
            summary["framework"] = framework
            summary["parsed"] = parsed
            if parsed is None:
                passed = False
            else:
                passed = passed and parsed["failed"] == 0 and parsed["passed"] > 0
        return passed, summary

    def _parse_framework_result(self, framework: str, stdout: str, stderr: str) -> dict[str, int] | None:
        text = "\n".join(part for part in [stdout, stderr] if part)
        if framework == "pytest":
            passed_match = re.search(r"(?P<passed>\d+)\s+passed", text)
            failed_match = re.search(r"(?P<failed>\d+)\s+failed", text)
            if passed_match is None and failed_match is None:
                return None
            return {
                "passed": int(passed_match.group("passed")) if passed_match else 0,
                "failed": int(failed_match.group("failed")) if failed_match else 0,
            }
        if framework == "unittest":
            ran_match = re.search(r"Ran\s+(?P<ran>\d+)\s+tests?", text)
            failed_match = re.search(r"FAILED\s+\((?:failures|errors)=(?P<count>\d+)", text)
            ok = "OK" in text
            if ran_match is None:
                return None
            failed = int(failed_match.group("count")) if failed_match else 0
            passed = int(ran_match.group("ran")) if ok and failed == 0 else max(int(ran_match.group("ran")) - failed, 0)
            return {"passed": passed, "failed": failed}
        return None

    def _run_structural_check(
        self,
        check: dict[str, Any],
        *,
        artifacts: VerificationArtifacts,
        plan: Plan | None = None,
        step: PlanStep | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        check_type = str(check.get("check_type", "")).strip()
        if check_type == "dependencies_completed":
            if plan is None or step is None:
                raise VerificationError("dependencies_completed check requires plan and step")
            completed = {item.step_id for item in plan.steps if item.status == "completed"}
            missing = [dependency for dependency in step.depends_on if dependency not in completed]
            return not missing, {"completed_dependencies": sorted(completed), "missing_dependencies": missing}
        if check_type == "file_exists":
            path = Path(str(check.get("path", ""))).expanduser()
            return path.exists(), {"path": str(path), "exists": path.exists()}
        if check_type == "file_contains":
            path = Path(str(check.get("path", ""))).expanduser()
            if not path.exists():
                return False, {"path": str(path), "exists": False}
            text = path.read_text(encoding="utf-8")
            pattern = str(check.get("pattern", ""))
            regex = bool(check.get("regex", False))
            matched = bool(re.search(pattern, text, re.MULTILINE)) if regex else pattern in text
            return matched, {"path": str(path), "exists": True, "matched": matched}
        if check_type == "json_schema_valid":
            schema = check.get("schema")
            if not isinstance(schema, dict):
                raise VerificationError("json_schema_valid check requires a schema object")
            actual = self._resolve_actual(check, artifacts=artifacts)
            try:
                _validate_schema_value(actual, schema, path="verification")
            except ToolValidationError as exc:
                return False, {"error": str(exc), "actual": actual, "schema": schema}
            return True, {"actual": actual, "schema": schema}
        if check_type in {"function_exists", "symbol_exists"}:
            path = Path(str(check.get("path", ""))).expanduser()
            symbol = str(check.get("symbol", "") or check.get("function_name", "")).strip()
            if not path.exists():
                return False, {"path": str(path), "exists": False, "symbol": symbol}
            module = ast.parse(path.read_text(encoding="utf-8"))
            found = False
            for node in ast.walk(module):
                if check_type == "function_exists" and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == symbol:
                    found = True
                    break
                if check_type == "symbol_exists":
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == symbol:
                        found = True
                        break
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name) and target.id == symbol:
                                found = True
                                break
            return found, {"path": str(path), "symbol": symbol, "found": found}
        if check_type == "tool_output_schema_valid":
            latest = artifacts.latest_tool_result
            if latest is None:
                return False, {"reason": "missing_tool_result"}
            return True, {"tool_name": latest.tool_name, "output": latest.output}
        raise VerificationError(f"Unsupported structural check type: {check_type}")

    def _run_value_check(self, check: dict[str, Any], *, artifacts: VerificationArtifacts) -> tuple[bool, dict[str, Any]]:
        check_type = str(check.get("check_type", "")).strip()
        if check_type == "artifact_present":
            artifact_name = str(check.get("artifact", "")).strip()
            actual = self._resolve_named_artifact(artifact_name, artifacts)
            present = actual is not None
            return present, {"artifact": artifact_name, "present": present}
        if check_type == "tool_name_equals":
            latest = artifacts.latest_tool_result
            actual = latest.tool_name if latest is not None else None
            expected = str(check.get("expected", ""))
            return actual == expected, {"actual": actual, "expected": expected}
        if check_type == "tool_output_nonempty":
            latest = artifacts.latest_tool_result
            output = latest.output if latest is not None else None
            passed = isinstance(output, dict) and bool(output)
            return passed, {"output": output}
        if check_type == "tool_files_changed":
            latest = artifacts.latest_tool_result
            output = latest.output if latest is not None else None
            if not isinstance(output, dict):
                return False, {"output": output}
            changed = []
            for key in ("created_files", "modified_files", "deleted_files"):
                value = output.get(key)
                if isinstance(value, list):
                    changed.extend(str(item) for item in value if str(item))
            return bool(changed), {"changed_files": changed}
        if check_type == "string_nonempty":
            actual = self._resolve_actual(check, artifacts=artifacts)
            passed = isinstance(actual, str) and bool(actual.strip())
            return passed, {"actual": actual}
        if check_type == "exact_match":
            actual = self._resolve_actual(check, artifacts=artifacts)
            expected = check.get("expected")
            return actual == expected, {"actual": actual, "expected": expected}
        if check_type == "numeric_tolerance":
            actual = self._resolve_actual(check, artifacts=artifacts)
            expected = float(check.get("expected"))
            tolerance = float(check.get("tolerance", 0.0))
            if not isinstance(actual, (int, float)) or isinstance(actual, bool):
                return False, {"actual": actual, "expected": expected, "tolerance": tolerance}
            return abs(float(actual) - expected) <= tolerance, {"actual": actual, "expected": expected, "tolerance": tolerance}
        if check_type == "string_match":
            actual = self._resolve_actual(check, artifacts=artifacts)
            expected = str(check.get("expected", ""))
            passed = isinstance(actual, str) and actual == expected
            return passed, {"actual": actual, "expected": expected}
        raise VerificationError(f"Unsupported value check type: {check_type}")

    def _run_llm_fallback(
        self,
        *,
        runtime,
        state: SessionState,
        step: PlanStep,
        artifacts: VerificationArtifacts,
        checks: list[dict[str, Any]],
    ) -> list[tuple[str, bool, dict[str, Any]]]:
        criteria_specs = [
            {
                "name": str(check.get("name", "")).strip(),
                "criterion": str(check.get("criterion", "")).strip(),
            }
            for check in checks
        ]
        if (
            not criteria_specs
            or any(not item["name"] for item in criteria_specs)
            or any(not item["criterion"] for item in criteria_specs)
        ):
            raise VerificationError(f"Step {step.step_id} requires explicit criteria for llm_fallback verification")
        payload = runtime._run_llm_verification(
            state,
            step=step,
            criteria=criteria_specs,
            assistant_text=artifacts.assistant_text,
            evidence={
                "tool_result": artifacts.latest_tool_result.output if artifacts.latest_tool_result is not None else None,
                "tool_name": artifacts.latest_tool_result.tool_name if artifacts.latest_tool_result is not None else None,
            },
        )
        results_by_name: dict[str, dict[str, Any]] = {}
        for item in payload.get("criteria", []):
            if not isinstance(item, dict):
                raise VerificationError("LLM verification returned a non-object criterion result")
            name = str(item.get("name", "")).strip()
            if not name:
                raise VerificationError("LLM verification returned a criterion without a name")
            if not isinstance(item.get("passed"), bool):
                raise VerificationError(f"LLM verification returned non-boolean result for {name}")
            if not isinstance(item.get("evidence"), str):
                raise VerificationError(f"LLM verification returned non-string evidence for {name}")
            results_by_name[name] = item
        outputs: list[tuple[str, bool, dict[str, Any]]] = []
        for check in checks:
            name = str(check.get("name", "")).strip()
            criterion = str(check.get("criterion", "")).strip()
            item = results_by_name.get(name)
            if item is None:
                outputs.append((name, False, {"criterion": criterion, "error": "missing_criterion_result"}))
                continue
            outputs.append((name, bool(item["passed"]), {"criterion": criterion, "evidence": item["evidence"]}))
        return outputs

    def _resolve_named_artifact(self, name: str, artifacts: VerificationArtifacts) -> Any:
        if name == "tool_result":
            return artifacts.latest_tool_result.output if artifacts.latest_tool_result is not None else None
        if name == "assistant_text":
            return artifacts.assistant_text
        if name == "tool_name":
            return artifacts.latest_tool_result.tool_name if artifacts.latest_tool_result is not None else None
        return artifacts.runtime_artifacts.get(name)

    def _resolve_actual(self, check: dict[str, Any], *, artifacts: VerificationArtifacts) -> Any:
        if "actual" in check:
            return check["actual"]
        actual_source = str(check.get("actual_source", "")).strip()
        if not actual_source:
            raise VerificationError("Verification check is missing actual_source")
        if actual_source == "assistant_text":
            return artifacts.assistant_text
        if actual_source == "tool_name":
            return artifacts.latest_tool_result.tool_name if artifacts.latest_tool_result is not None else None
        if actual_source == "tool_output":
            return artifacts.latest_tool_result.output if artifacts.latest_tool_result is not None else None
        if actual_source.startswith("tool_output."):
            value: Any = artifacts.latest_tool_result.output if artifacts.latest_tool_result is not None else None
            for part in actual_source.split(".")[1:]:
                if not isinstance(value, dict) or part not in value:
                    return None
                value = value[part]
            return value
        return artifacts.runtime_artifacts.get(actual_source)


@dataclass(slots=True)
class BenchmarkVerificationReport:
    passed: bool
    task_type: str
    checks: dict[str, bool]
    evidence: dict[str, Any]
    reason: str


def verify_benchmark_contract(
    contract,
    *,
    assistant_text: str,
    state: SessionState,
    events: list[HistoryEvent],
    workspace_before: dict[str, str] | None = None,
    workspace_after: dict[str, str] | None = None,
) -> BenchmarkVerificationReport:
    checks: dict[str, bool] = {}
    evidence: dict[str, Any] = {}
    engine = VerificationEngine(semantic_backend_mode="degraded_lexical")

    required_events = set(getattr(contract, "required_history_events", []) or [])
    forbidden_events = set(getattr(contract, "forbidden_history_events", []) or [])
    event_counts = Counter(event.event_type for event in events)
    seen_events = set(event_counts)
    checks["required_history_events"] = required_events.issubset(seen_events)
    evidence["required_history_events"] = {
        "required": sorted(required_events),
        "seen": sorted(seen_events),
        "missing": sorted(required_events - seen_events),
    }
    if forbidden_events:
        checks["forbidden_history_events"] = forbidden_events.isdisjoint(seen_events)
        evidence["forbidden_history_events"] = {
            "forbidden": sorted(forbidden_events),
            "seen": sorted(seen_events & forbidden_events),
        }
    required_event_counts = dict(getattr(contract, "required_event_counts", {}) or {})
    if required_event_counts:
        checks["required_event_counts"] = all(event_counts.get(name, 0) >= count for name, count in required_event_counts.items())
        evidence["required_event_counts"] = {
            name: {"required": count, "actual": event_counts.get(name, 0)}
            for name, count in sorted(required_event_counts.items())
        }

    expected_answer = getattr(contract, "expected_answer", None)
    if expected_answer is not None:
        checks["expected_answer"] = assistant_text.strip() == str(expected_answer).strip()
        evidence["expected_answer"] = {"actual": assistant_text.strip(), "expected": str(expected_answer).strip()}
    expected_answer_contains = list(getattr(contract, "expected_answer_contains", []) or [])
    if expected_answer_contains:
        checks["expected_answer_contains"] = all(fragment in assistant_text for fragment in expected_answer_contains)
        evidence["expected_answer_contains"] = {
            "actual": assistant_text,
            "required_fragments": expected_answer_contains,
        }
    expected_answer_regex = getattr(contract, "expected_answer_regex", None)
    if expected_answer_regex is not None:
        checks["expected_answer_regex"] = re.search(str(expected_answer_regex), assistant_text) is not None
        evidence["expected_answer_regex"] = {"actual": assistant_text, "pattern": str(expected_answer_regex)}

    expected_files = dict(getattr(contract, "expected_files", {}) or {})
    if expected_files:
        file_results: dict[str, bool] = {}
        file_evidence: dict[str, Any] = {}
        for path_text, expected_content in expected_files.items():
            key = str(path_text)
            path = Path(key)
            exists = path.exists()
            actual = path.read_text(encoding="utf-8") if exists else None
            file_results[key] = exists and actual == expected_content
            file_evidence[key] = {"exists": exists, "actual": actual, "expected": expected_content}
        checks["expected_files"] = all(file_results.values())
        evidence["expected_files"] = file_evidence
    expected_file_patterns = dict(getattr(contract, "expected_file_patterns", {}) or {})
    if expected_file_patterns:
        pattern_results: dict[str, bool] = {}
        pattern_evidence: dict[str, Any] = {}
        for path_text, patterns in expected_file_patterns.items():
            key = str(path_text)
            path = Path(key)
            exists = path.exists()
            actual = path.read_text(encoding="utf-8") if exists else None
            matches = exists and all(pattern in actual for pattern in patterns)
            pattern_results[key] = matches
            pattern_evidence[key] = {"exists": exists, "patterns": list(patterns), "actual": actual}
        checks["expected_file_patterns"] = all(pattern_results.values())
        evidence["expected_file_patterns"] = pattern_evidence

    command = list(getattr(contract, "command", []) or [])
    if command:
        command_check = {
            "check_type": "command_success",
            "command": command,
            "cwd": getattr(contract, "command_cwd", None),
            "framework": getattr(contract, "command_framework", None),
        }
        command_passed, command_evidence = engine._run_execution_check(command_check)
        checks["command"] = command_passed
        evidence["command"] = command_evidence

    tool_calls = [event for event in events if event.event_type == "tool_called"]
    tool_names = [str(event.payload.get("tool_name", "")) for event in tool_calls]
    required_tools = list(getattr(contract, "required_tools_used", []) or [])
    if required_tools:
        checks["required_tools_used"] = all(tool in tool_names for tool in required_tools)
        evidence["required_tools_used"] = {"required": required_tools, "actual": tool_names}
    forbidden_tools = list(getattr(contract, "forbidden_tools_used", []) or [])
    if forbidden_tools:
        checks["forbidden_tools_used"] = all(tool not in tool_names for tool in forbidden_tools)
        evidence["forbidden_tools_used"] = {"forbidden": forbidden_tools, "actual": tool_names}
    min_tool_calls = getattr(contract, "min_tool_calls", None)
    if min_tool_calls is not None:
        checks["min_tool_calls"] = len(tool_calls) >= int(min_tool_calls)
        evidence["min_tool_calls"] = {"required": int(min_tool_calls), "actual": len(tool_calls)}
    max_tool_calls = getattr(contract, "max_tool_calls", None)
    if max_tool_calls is not None:
        checks["max_tool_calls"] = len(tool_calls) <= int(max_tool_calls)
        evidence["max_tool_calls"] = {"required": int(max_tool_calls), "actual": len(tool_calls)}
    expected_stop_reason = getattr(contract, "expected_stop_reason", None)
    if expected_stop_reason is not None:
        actual_reason = state.metrics.last_reasoning_reason
        checks["expected_stop_reason"] = actual_reason == str(expected_stop_reason)
        evidence["expected_stop_reason"] = {"expected": str(expected_stop_reason), "actual": actual_reason}

    if workspace_after is not None:
        before_snapshot = dict(workspace_before or {})
        after_snapshot = dict(workspace_after)
        snapshot_keys = set(before_snapshot) | set(after_snapshot)

        def _normalize_workspace_key(path_text: str) -> str:
            path_text = str(path_text)
            if path_text in snapshot_keys:
                return path_text
            for key in snapshot_keys:
                if path_text.endswith(key):
                    return key
            return path_text

        changed_files = sorted(
            path
            for path in set(before_snapshot) | set(after_snapshot)
            if before_snapshot.get(path) != after_snapshot.get(path)
        )
        evidence["workspace_changes"] = {
            "changed_files": changed_files,
            "before_count": len(before_snapshot),
            "after_count": len(after_snapshot),
        }
        allowed_modified_files = [_normalize_workspace_key(str(path)) for path in getattr(contract, "allowed_modified_files", []) or []]
        if allowed_modified_files:
            allowed_with_backups = set(allowed_modified_files)
            allowed_with_backups.update(f"{path}.bak" for path in allowed_modified_files)
        else:
            allowed_with_backups = set()
        if allowed_modified_files:
            checks["allowed_modified_files"] = set(changed_files).issubset(allowed_with_backups)
            evidence["allowed_modified_files"] = {
                "allowed": sorted(allowed_with_backups),
                "changed_files": changed_files,
            }
        elif getattr(contract, "forbid_unexpected_workspace_changes", False):
            expected_changed = {_normalize_workspace_key(path) for path in set(expected_files) | set(expected_file_patterns)}
            expected_with_backups = set(expected_changed)
            expected_with_backups.update(f"{path}.bak" for path in expected_changed)
            checks["allowed_modified_files"] = set(changed_files).issubset(expected_with_backups)
            evidence["allowed_modified_files"] = {
                "allowed": sorted(expected_with_backups),
                "changed_files": changed_files,
            }

    task_type = str(getattr(contract, "task_type", ""))
    policy_checks: list[bool] = []
    if task_type == "coding":
        policy_checks.append(bool(command or expected_files or expected_file_patterns))
    elif task_type == "file_edit":
        policy_checks.append(bool(expected_files or expected_file_patterns))
    elif task_type == "reading":
        policy_checks.append(bool(expected_answer is not None or expected_answer_contains or getattr(contract, "expected_json", None) is not None))
    elif task_type == "multi_step":
        policy_checks.append(bool(command or expected_files or required_events))
    elif task_type == "failure":
        policy_checks.append(bool(required_events or getattr(contract, "expected_stop_reason", None)))
    elif task_type == "quality":
        policy_checks.append(bool(expected_answer is not None or expected_answer_contains or required_events))
    if policy_checks:
        checks["verifier_policy_complete"] = all(policy_checks)
        evidence["verifier_policy_complete"] = {"task_type": task_type, "checks": policy_checks}

    expected_json = getattr(contract, "expected_json", None)
    expected_schema = getattr(contract, "expected_json_schema", None)
    if expected_json is not None or expected_schema is not None:
        try:
            parsed = json.loads(assistant_text)
            evidence["reading_json"] = {"parsed": parsed}
            checks["reading_json_parse"] = True
        except json.JSONDecodeError as exc:
            parsed = None
            checks["reading_json_parse"] = False
            evidence["reading_json"] = {"error": str(exc), "text": assistant_text}
        if expected_json is not None:
            checks["reading_json_exact"] = parsed == expected_json
            evidence["reading_json_exact"] = {"actual": parsed, "expected": expected_json}
        if expected_schema is not None:
            try:
                _validate_schema_value(parsed, expected_schema, path="benchmark.reading")
            except ToolValidationError as exc:
                checks["reading_json_schema"] = False
                evidence["reading_json_schema"] = {"error": str(exc), "schema": expected_schema, "actual": parsed}
            else:
                checks["reading_json_schema"] = True
                evidence["reading_json_schema"] = {"schema": expected_schema, "actual": parsed}

    if task_type == "failure":
        failure_signals = bool(state.metrics.steps_failed or state.metrics.verification_failures or state.metrics.failed_turns)
        abnormal_stop = state.metrics.last_reasoning_reason not in {"answered", "final_response"}
        checks["failure_signals"] = failure_signals or abnormal_stop or checks.get("expected_answer", False)
        evidence["failure_signals"] = {
            "steps_failed": state.metrics.steps_failed,
            "verification_failures": state.metrics.verification_failures,
            "failed_turns": state.metrics.failed_turns,
            "last_reasoning_reason": state.metrics.last_reasoning_reason,
        }

    passed = all(checks.values()) if checks else False
    if passed:
        reason = "benchmark_contract_passed"
    else:
        failing = [name for name, ok in checks.items() if not ok]
        reason = ",".join(failing) if failing else "benchmark_contract_failed"
    return BenchmarkVerificationReport(
        passed=passed,
        task_type=task_type or "unknown",
        checks=checks,
        evidence=evidence,
        reason=reason,
    )
