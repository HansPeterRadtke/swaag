from __future__ import annotations

from dataclasses import dataclass, field

from swaag.evaluator import EvaluationOutcome
from swaag.types import BudgetReport, ToolExecutionResult


@dataclass(slots=True)
class SubsystemExecutionResult:
    subsystem_name: str
    success: bool
    progress: list[str] = field(default_factory=list)
    tool_results: list[ToolExecutionResult] = field(default_factory=list)
    budget_reports: list[BudgetReport] = field(default_factory=list)
    assistant_text: str = ""
    evaluation: EvaluationOutcome | None = None
    background_job_started: bool = False
    background_process_id: str | None = None
