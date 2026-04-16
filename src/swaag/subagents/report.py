from __future__ import annotations

from swaag.subagents.protocol import SubagentReport


def report_summary(report: SubagentReport) -> str:
    status = "accepted" if report.accepted else "rejected"
    return f"{report.spec.subagent_type}:{status}:{report.reason}"
