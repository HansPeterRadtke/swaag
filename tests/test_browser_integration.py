from __future__ import annotations

import json
import sys
from pathlib import Path

from swaag.runtime import AgentRuntime

from tests.helpers import FakeModelClient, plan_response, plan_step


def _write_fake_aubro_script(tmp_path: Path) -> Path:
    script = tmp_path / "fake_aubro.py"
    script.write_text(
        """
from __future__ import annotations

import json
import sys


def _flag(name: str, default: str) -> str:
    if name not in sys.argv:
        return default
    return sys.argv[sys.argv.index(name) + 1]


def main() -> int:
    if len(sys.argv) < 2:
        return 2
    command = sys.argv[1]
    if command == "search":
        query = sys.argv[2]
        if query == "badjson":
            print("not-json")
            return 0
        engine = _flag("--engine", "auto")
        limit = int(_flag("--limit", "5"))
        payload = {
            "query": query,
            "engine": engine,
            "url": "https://example.test/search",
            "results": [
                {
                    "title": f"Result {index}",
                    "url": f"https://example.test/r/{index}",
                    "snippet": "x" * 80,
                }
                for index in range(limit + 2)
            ],
            "attempts": [
                {
                    "engine": engine,
                    "url": "https://example.test/search",
                    "results": limit + 2,
                    "blocked": False,
                }
            ],
        }
        print(json.dumps(payload))
        return 0
    if command == "browse":
        url = sys.argv[2]
        payload = {
            "url": url,
            "title": "Example page",
            "backend": "playwright",
            "blocked": False,
            "block_reason": None,
            "text": "page-text-" * 800,
            "links": [
                {"text": f"Link {index}", "href": f"https://example.test/link/{index}"}
                for index in range(7)
            ],
            "forms": [{"name": "login"}, {"name": "search"}],
            "buttons": [{"text": "Save"}, {"text": "Submit"}, {"text": "Cancel"}],
        }
        print(json.dumps(payload))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
        """.strip()
        + "\n",
        encoding="utf-8",
    )
    return script


def _browser_config(make_config, script: Path):
    return make_config(
        tools__enabled=[
            "echo",
            "time_now",
            "calculator",
            "list_files",
            "read_file",
            "read_text",
            "notes",
            "edit_text",
            "write_file",
            "run_tests",
            "shell_command",
            "browser_search",
            "browser_browse",
        ],
        tools__allow_stateful_tools=True,
        environment__aubro_entrypoint=f"{sys.executable} {script}",
        environment__aubro_timeout_seconds=15,
        environment__aubro_max_text_chars=24,
        environment__aubro_max_results=2,
        environment__aubro_max_links=3,
    )


def test_browser_search_tool_runs_through_aubro_bridge(make_config, tmp_path: Path) -> None:
    config = _browser_config(make_config, _write_fake_aubro_script(tmp_path))
    runtime = AgentRuntime(config, model_client=FakeModelClient())

    run = runtime.execute_tool_once(
        "browser_search",
        {"query": "browser automation docs", "engine": "duckduckgo", "limit": 9},
    )

    assert run.tool_result is not None
    assert run.tool_result.output == {
        "query": "browser automation docs",
        "engine": "duckduckgo",
        "url": "https://example.test/search",
        "result_count": 2,
        "results": [
            {
                "title": "Result 0",
                "url": "https://example.test/r/0",
                "snippet": "x" * 24,
            },
            {
                "title": "Result 1",
                "url": "https://example.test/r/1",
                "snippet": "x" * 24,
            },
        ],
        "attempts": [
            {
                "engine": "duckduckgo",
                "url": "https://example.test/search",
                "results": 4,
                "blocked": False,
            }
        ],
    }

    events = runtime.history.read_history(run.session_id)
    event_types = [event.event_type for event in events]
    assert event_types.count("process_started") == 1
    assert event_types.count("process_completed") == 1
    assert event_types.count("tool_called") == 1
    assert event_types.count("tool_result") == 1


def test_browser_browse_tool_returns_trimmed_page_summary(make_config, tmp_path: Path) -> None:
    config = _browser_config(make_config, _write_fake_aubro_script(tmp_path))
    runtime = AgentRuntime(config, model_client=FakeModelClient())

    run = runtime.execute_tool_once(
        "browser_browse",
        {"url": "https://example.test/docs"},
    )

    assert run.tool_result is not None
    assert run.tool_result.output == {
        "url": "https://example.test/docs",
        "title": "Example page",
        "backend": "playwright",
        "blocked": False,
        "block_reason": "",
        "text_excerpt": ("page-text-" * 800)[:24],
        "link_count": 7,
        "links": [
            {"text": "Link 0", "href": "https://example.test/link/0"},
            {"text": "Link 1", "href": "https://example.test/link/1"},
            {"text": "Link 2", "href": "https://example.test/link/2"},
        ],
        "form_count": 2,
        "button_count": 3,
    }


def test_browser_tool_reports_error_when_aubro_returns_invalid_json(make_config, tmp_path: Path) -> None:
    config = _browser_config(make_config, _write_fake_aubro_script(tmp_path))
    runtime = AgentRuntime(config, model_client=FakeModelClient())

    run = runtime.execute_tool_once(
        "browser_search",
        {"query": "badjson", "engine": "auto", "limit": 1},
    )

    assert run.tool_result is None
    events = runtime.history.read_history(run.session_id)
    tool_error = next(event for event in events if event.event_type == "tool_error")
    assert tool_error.payload["tool_name"] == "browser_search"
    assert tool_error.payload["error_type"] == "BrowserAutomationError"


def test_runtime_run_turn_executes_browser_search_end_to_end(make_config, tmp_path: Path) -> None:
    # Goal text explicitly names the tool so the literal-name parser
    # in _required_tools_for_goal recognizes it. The previous version
    # relied on a removed heuristic that mapped "Search the web" →
    # browser_search.
    goal = (
        "Use browser_search to find browser automation docs. Reply exactly done."
    )
    config = _browser_config(make_config, _write_fake_aubro_script(tmp_path))
    fake_client = FakeModelClient(
        responses=[
            plan_response(
                goal=goal,
                steps=[
                    plan_step(
                        "step_search",
                        "Search the web for browser automation docs",
                        "tool",
                        expected_tool="browser_search",
                        expected_output="Browser search result",
                        success_criteria="The search returns results.",
                    ),
                    plan_step(
                        "step_answer",
                        "Reply to the user",
                        "respond",
                        expected_output="done",
                        success_criteria="The user receives the exact reply.",
                        depends_on=["step_search"],
                    ),
                ],
            ),
            json.dumps(
                {
                    "action": "call_tool",
                    "response": "",
                    "tool_name": "browser_search",
                    "tool_input": {
                        "query": "browser automation docs",
                        "engine": "auto",
                        "limit": 2,
                    },
                }
            ),
            "done",
        ]
    )
    runtime = AgentRuntime(config, model_client=fake_client)

    result = runtime.run_turn(goal)

    assert result.assistant_text == "done"
    assert [item.tool_name for item in result.tool_results] == ["browser_search"]
    events = runtime.history.read_history(result.session_id)
    assert any(event.event_type == "verification_passed" for event in events)
