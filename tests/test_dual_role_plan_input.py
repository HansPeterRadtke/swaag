"""Regression from an observed live Coder->Q8 orchestration rejection.

The planner duplicated node completion_mode='natural' at both levels of its
plan.apply call. This is an identical redundant value, not a new instruction.
"""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from swaag.tools.orchestration import OrchestrationControlTool
from swaag.tools.base import ToolValidationError


def _plan_args(*, top_mode="natural", node_modes=("natural",)):
    tool = OrchestrationControlTool()
    args = {key: None for key in tool.input_schema["required"]}
    args.update({
        "operation": "plan.apply",
        "completion_mode": top_mode,
        "objective": "Verify an isolated background calculation",
        "node_id": "worker42",
        "plan_spec": {
            "objective": "Verify an isolated background calculation",
            "scheduling_mode": "sequential",
            "max_parallel": 1,
            "reporting_mode": "completion_only",
            "resource_limits_json": "{}",
            "nodes": [
                {"key": f"worker{idx}", "completion_mode": mode,
                 "objective": f"Return marker {idx}", "priority": 1,
                 "model_key": "default", "finish_criteria": "result delivered",
                 "abort_criteria": "failure", "resources_json": "{}"}
                for idx, mode in enumerate(node_modes)
            ],
            "edges": [],
            "start": True,
        },
    })
    return args


def test_duplicate_plan_mode_from_live_model_is_noop_when_every_node_agrees():
    tool = OrchestrationControlTool()
    args = _plan_args()
    input_copy = deepcopy(args)
    valid = tool.validate(args)
    assert args == input_copy  # Never silently mutate the model's retained audit payload.
    assert valid["completion_mode"] is None
    assert valid["operation"] == "plan.apply"
    assert valid["plan_spec"]["nodes"][0]["completion_mode"] == "natural"

    class FakeApi:
        def __init__(self): self.calls = []
        def execute(self, operation, payload):
            self.calls.append((operation, deepcopy(payload)))
            return {"ok": True, "plan_id": "plan_test"}

    fake = FakeApi()
    context = SimpleNamespace(runtime_capabilities={"orchestration": fake})
    result = tool.execute(valid, context)
    assert fake.calls and fake.calls[0][0] == "plan.apply"
    assert "completion_mode" not in fake.calls[0][1]
    assert fake.calls[0][1]["plan_spec"]["nodes"][0]["completion_mode"] == "natural"


@pytest.mark.parametrize("node_modes", [("continuous",), ("natural", "continuous")])
def test_conflicting_plan_modes_still_fail_closed(node_modes):
    with pytest.raises(ToolValidationError, match="completion_mode"):
        OrchestrationControlTool().validate(_plan_args(node_modes=node_modes))


def test_unrelated_operation_cannot_use_node_completion_mode():
    args = _plan_args()
    args["operation"] = "create"
    with pytest.raises(ToolValidationError, match="completion_mode"):
        OrchestrationControlTool().validate(args)


def test_plan_mode_none_does_not_modify_explicit_node_modes():
    args = _plan_args(top_mode=None, node_modes=("natural", "continuous"))
    valid = OrchestrationControlTool().validate(args)
    assert [n["completion_mode"] for n in valid["plan_spec"]["nodes"]] == ["natural", "continuous"]


def test_plan_apply_empty_node_list_never_accepted():
    args = _plan_args()
    args["plan_spec"]["nodes"] = []
    with pytest.raises(ToolValidationError, match="non-empty"):
        OrchestrationControlTool().validate(args)
