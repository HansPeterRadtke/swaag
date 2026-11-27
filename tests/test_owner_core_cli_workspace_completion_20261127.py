"""Owner-verified core SWAAG CLI/orchestrator failure reproductions, all offline.

Actual Nitro incident: launching from /data/infra/bin made sibling libexec
inaccessible; model searched wrong directory then repeated tool errors.
Never operate on the real user's active session or call a resident model.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from swaag.cli import main
from swaag.config import load_config
from swaag.history import HistoryStore
from swaag.environment.environment import AgentEnvironment
from swaag.runtime import AgentRuntime
from swaag.tools.registry import ToolRegistry
from tests.test_agent_action_loop import FakeModelClient, _action


def _model_runtime(config, responses):
    model = FakeModelClient(list(responses))
    return AgentRuntime(config, model_client=model), model


def _user_session(config):
    return HistoryStore(config.sessions.root).create(
        config_fingerprint=config.config_fingerprint(),
        model_base_url=config.model.base_url,
    )


def test_actual_requested_allowed_directory_lists_bounded_entries(make_config, tmp_path):
    folder = tmp_path / "libexec"
    folder.mkdir()
    for index in range(5):
        (folder / f"chatgpt_{index}.sh").write_text("# test-only\n")
    config = make_config(agent_data__max_list_files=3)
    state = _user_session(config)
    tool, result = ToolRegistry().dispatch("list_files", {"path": str(folder)}, config, state)
    assert result.tool_name == "list_files"
    assert result.output["count"] == 3
    assert result.output["truncated"] is True
    assert set(result.output) == {"path", "entries", "count", "truncated"}
    assert all(name.startswith("libexec/chatgpt_") for name in result.output["entries"])
    assert len(result.generated_events) == 1
    # The same tool retains a truthful nontruncated result for smaller scope.
    config.agent_data.max_list_files = 10
    _, full = ToolRegistry().dispatch("list_files", {"path": str(folder)}, config, state)
    assert full.output["count"] == 5
    assert full.output["truncated"] is False


def test_forbidden_exact_user_absolute_path_returns_visible_blocker_without_fallback(make_config, tmp_path):
    outside = tmp_path.parent / "not-authorized" / "libexec"
    config = make_config(runtime__max_identical_tool_failures=2, model__context_limit=32000)
    config.tools.read_roots = [tmp_path]
    runtime, model = _model_runtime(config, [
        _action(message="looking", tool_calls=[("list_files", {"path": str(outside)})], continue_loop=True),
    ])
    result = runtime.run_turn(f"Inspect chatgpt scripts in {outside}.")
    assert "could not inspect" in result.assistant_text.lower()
    assert str(outside) in result.assistant_text
    assert str(tmp_path) in result.assistant_text
    assert "not inspected" in result.assistant_text
    assert "--root" in result.assistant_text
    assert len(model.requests) == 1
    history = runtime.history.read_history(result.session_id)
    assert [e.payload.get("tool_name") for e in history if e.event_type == "tool_called"] == ["list_files"]
    assert sum(e.event_type == "tool_error" for e in history) == 1
    assert history[-1].event_type == "turn_finished"


def test_identical_non_authoritative_error_stops_after_configured_repetition(make_config, tmp_path):
    missing = "not-created.txt"  # Not part of the original request text.
    config = make_config(runtime__max_identical_tool_failures=2, model__context_limit=32000)
    action = _action(tool_calls=[("read_file", {"path": missing})], continue_loop=True)
    runtime, model = _model_runtime(config, [action, action])
    result = runtime.run_turn("Check a missing file and tell me why it could not be read.")
    assert "stopped after 2 identical failed read_file calls" in result.assistant_text
    assert "File does not exist" in result.assistant_text
    assert len(model.requests) == 2
    events = runtime.history.read_history(result.session_id)
    assert sum(e.event_type == "tool_error" for e in events) == 2
    assert sum(e.event_type == "turn_finished" for e in events) == 1


def test_repeated_incomplete_evaluator_is_bounded_without_fabricated_success(make_config, monkeypatch):
    config = make_config(runtime__completion_evaluation_enabled=True,
                         runtime__max_completion_rejections=2, model__context_limit=32000)
    runtime, model = _model_runtime(config, [
        _action(message="The requested output is available."),
        _action(message="The requested output is available."),
    ])
    invocations = []
    def incomplete(state, **kwargs):
        invocations.append(kwargs["original_request"])
        runtime.history.record_event(state, "completion_evaluated", {
            "complete": False, "reason": "Missing exact requested file evidence", "remaining_work": ["Inspect original directory"]})
        return {"complete":False,"reason":"Missing exact requested file evidence","remaining_work":["Inspect original directory"]}
    monkeypatch.setattr(runtime, "_evaluate_completion", incomplete)
    result = runtime.run_turn("Inspect the exact directory and then answer.")
    assert "stopped because 2 completion evaluations" in result.assistant_text
    assert "task remains incomplete" in result.assistant_text
    assert "Missing exact requested file evidence" in result.assistant_text
    assert invocations == ["Inspect the exact directory and then answer."] * 2
    assert len(model.requests) == 2
    events = runtime.history.read_history(result.session_id)
    assert sum(e.event_type == "completion_evaluated" for e in events) == 2
    rejections=[e.payload for e in events if e.event_type == "completion_rejected"]
    assert [e["attempt"] for e in rejections] == [1, 2]
    assert [e["retry_budget_exhausted"] for e in rejections] == [False, True]
    assert all(e["max_attempts"] == 2 for e in rejections)
    assert sum(e.event_type == "turn_finished" for e in events) == 1


def test_action_budget_caps_successful_but_nonterminal_repetition(make_config):
    config = make_config(runtime__max_turn_actions=2, model__context_limit=32000)
    step = _action(tool_calls=[("calculator", {"expression":"1+1"})], continue_loop=True)
    runtime, model = _model_runtime(config, [step, step])
    result = runtime.run_turn("Keep checking this calculation until completion.")
    assert "bounded action budget was exhausted" in result.assistant_text
    assert "not complete" in result.assistant_text
    assert len(model.requests) == 2
    assert len(result.tool_results) == 2
    assert [x.output["result"] for x in result.tool_results] == [2,2]


def test_chat_auto_workspace_git_top_level_and_separate_cwd(monkeypatch, make_config, tmp_path, capsys):
    repo = tmp_path / "owner-project"
    (repo / "bin").mkdir(parents=True)
    (repo / "libexec").mkdir()
    (repo / "libexec" / "chatgpt_test.sh").write_text("test")
    subprocess.run(["git","init","-q",str(repo)],check=True)
    config = make_config()
    config.tools.read_roots = [Path(".")]
    roots = []
    class FakeService:
        def __init__(self, runtime):
            roots.append(str(runtime.config.tools.read_roots[0]))
            self.orchestrator_runtime = runtime
        def iter_orchestrator_message(self, prompt):
            yield {"event":"completed", "result":{"answer":"No model invoked"}}
        def close(self):pass
    monkeypatch.setattr("swaag.cli.load_config", lambda _: config)
    monkeypatch.setattr("swaag.cli.CommunicationService.from_runtime", lambda runtime: FakeService(runtime))
    monkeypatch.setattr("builtins.input", lambda prompt: "/quit")
    monkeypatch.chdir(repo / "bin")
    assert main(["chat"]) == 0
    assert roots == [str(repo)]
    assert (repo / "libexec").is_relative_to(Path(roots[0]))
    # A different non-Git workspace must not silently gain access to repo.
    other = tmp_path / "scratch";other.mkdir()
    config.tools.read_roots = [Path(".")]
    monkeypatch.chdir(other)
    assert main(["chat"]) == 0
    assert roots[-1] == str(other)
    assert not (repo / "libexec").is_relative_to(Path(roots[-1]))


def test_chat_explicit_root_and_invalid_root(monkeypatch, make_config, tmp_path):
    project = tmp_path / "project";project.mkdir()
    other=tmp_path / "scratch";other.mkdir()
    config = make_config()
    config.tools.read_roots = [other]
    roots=[]
    class FakeService:
        def __init__(self, runtime):
            self.orchestrator_runtime=runtime
            roots.append(str(runtime.config.tools.read_roots[0]))
        def close(self):pass
    monkeypatch.setattr("swaag.cli.load_config",lambda _:config)
    monkeypatch.setattr("swaag.cli.CommunicationService.from_runtime", lambda runtime:FakeService(runtime))
    monkeypatch.setattr("builtins.input",lambda _:"/quit")
    monkeypatch.chdir(other)
    assert main(["chat","--root",str(project)])==0
    assert roots==[str(project)]
    with pytest.raises(SystemExit):main(["chat","--root",str(tmp_path / "missing")])


def test_resume_rebases_existing_workspace_through_durable_history(make_config, tmp_path):
    old = tmp_path / "old";new=tmp_path / "new";old.mkdir();new.mkdir()
    config = make_config();config.tools.read_roots=[old]
    runtime=AgentRuntime(config,model_client=FakeModelClient([]))
    state=runtime.create_or_load_user_session("SWAAG Orchestrator")
    assert state.environment.workspace.root == str(old)
    runtime.config.tools.read_roots=[new]
    runtime._ensure_environment_initialized(state)
    assert state.environment.workspace.root == str(new)
    assert state.environment.workspace.cwd == str(new)
    reread=runtime.history.rebuild_from_history(state.session_id)
    assert reread.environment.workspace.root == str(new)
    assert sum(e.event_type=="environment_initialized" for e in runtime.history.read_history(state.session_id)) == 2


@pytest.mark.parametrize("setting", ["MAX_TURN_ACTIONS", "MAX_IDENTICAL_TOOL_FAILURES", "MAX_COMPLETION_REJECTIONS"])
def test_runtime_mechanical_budgets_reject_nonpositive_configuration(setting, tmp_path):
    with pytest.raises(ValueError, match=setting.lower()):
        load_config(env={
            "SWAAG__SESSIONS__ROOT": str(tmp_path / "sessions"),
            "SWAAG__RUNTIME__" + setting: "0",
        })


def test_live_evaluator_contract_can_reject_repeatedly_without_infinite_calls(make_config):
    """Real completion evaluator and constrained model grammar, deterministic fake client."""
    from swaag.types import CompletionResult

    class ReviewerClient(FakeModelClient):
        def __init__(self):
            super().__init__([])
            self.categories=[]
        def send_completion(self, payload, **kwargs):
            schema=payload.get("json_schema",{})
            fields=schema.get("properties",{})
            if "complete" in fields:
                self.categories.append("verdict")
                result={"complete":False,"reason":"The requested evidence was not inspected.","remaining_work":["Read the required file"]}
                if "evidence_requests" in fields:result["evidence_requests"]=[]
                answer=json.dumps(result)
            elif "tool_calls" in fields:
                self.categories.append("action")
                answer=_action(message="It was fully checked.")
            else:
                raise AssertionError("Unexpected verifier dependency: "+str(payload.get("contract")))
            self.requests.append(payload)
            return CompletionResult(text=answer,raw_request=payload,raw_response={"content":answer},prompt_tokens=None,completion_tokens=None,finish_reason="stop")
    cfg=make_config(model__context_limit=32000,
                    runtime__completion_evaluation_enabled=True,
                    runtime__max_completion_rejections=2)
    model=ReviewerClient()
    runtime=AgentRuntime(cfg,model_client=model)
    result=runtime.run_turn("Inspect the requested file and verify the result.")
    assert "task remains incomplete" in result.assistant_text
    assert "The requested evidence was not inspected." in result.assistant_text
    assert model.categories == ["action", "verdict", "action", "verdict"]
    events=runtime.history.read_history(result.session_id)
    rejected=[e.payload for e in events if e.event_type=="completion_rejected"]
    assert len(rejected)==2
    assert [e["attempt"] for e in rejected]==[1,2]
    assert rejected[-1]["retry_budget_exhausted"]
    assert sum(e.event_type=="turn_finished" for e in events)==1


def test_workspace_root_selection_updates_durable_config_fingerprint(make_config, tmp_path, monkeypatch):
    import swaag.cli as cli
    a=tmp_path/"alpha";b=tmp_path/"beta";a.mkdir();b.mkdir()
    config=make_config()
    config.tools.read_roots=[Path(".")]
    original=config.config_fingerprint()
    recorded=[]
    class FakeRuntime:
        def __init__(self,config):
            self.config=config
            recorded.append((config.config_fingerprint(),config.raw["tools"]["read_roots"]))
        def create_or_load_user_session(self, ref):
            assert ref == "SWAAG Orchestrator"
            return type("State", (), {"session_id":"source-only-test","session_name":ref})()
    class FakeService:
        def __init__(self,runtime):self.orchestrator_runtime=runtime
        def close(self):pass
    monkeypatch.setattr(cli,"load_config",lambda _:config)
    monkeypatch.setattr(cli,"AgentRuntime",FakeRuntime)
    monkeypatch.setattr(cli.CommunicationService,"from_runtime",lambda runtime:FakeService(runtime))
    monkeypatch.setattr("builtins.input",lambda _:"/quit")
    monkeypatch.chdir(a)
    assert cli.main(["chat"])==0
    config.tools.read_roots=[Path(".")]
    monkeypatch.chdir(b)
    assert cli.main(["chat"])==0
    assert recorded[0][1]==[str(a)] and recorded[1][1]==[str(b)]
    assert recorded[0][0]!=recorded[1][0]!=original


def test_one_slot_model_timeout_exits_cli_with_visible_error_not_false_completed(monkeypatch,make_config,capsys):
    """Transport exhaustion must not be mislabeled as a completed answer."""
    import swaag.cli as cli
    from unittest.mock import Mock
    config=make_config()
    calls=[]
    class FakeRuntime:
        def __init__(self, config):
            self.config=config
    class Service:
        def iter_orchestrator_message(self, message):
            calls.append(message)
            yield {"event":"accepted","session_id":"isolated","control_id":"inert"}
            raise TimeoutError("Coder has one occupied inference slot; no answer returned")
        def close(self):
            calls.append("closed")
    monkeypatch.setattr(cli,"load_config",lambda _:config)
    monkeypatch.setattr(cli,"AgentRuntime",FakeRuntime)
    monkeypatch.setattr(cli.CommunicationService,"from_runtime",lambda runtime:Service())
    monkeypatch.setattr(cli,"configure_operations_log",lambda *args,**kwargs:Mock())
    rc=cli.main(["ask","Tell me which scripts are in the requested directory."])
    observed=capsys.readouterr()
    assert rc==1
    assert "ERROR: Coder has one occupied inference slot" in observed.err
    assert "completed" not in observed.out.lower()
    assert calls==["Tell me which scripts are in the requested directory.","closed"]


def test_current_root_authority_is_explicit_in_prompt_context(make_config, tmp_path):
    """Historical workspace failures cannot masquerade as current authority."""
    old = tmp_path / "old"; new = tmp_path / "new"
    old.mkdir(); new.mkdir()
    config = make_config()
    config.tools.read_roots = [old]
    runtime = AgentRuntime(config, model_client=FakeModelClient([]))
    state = runtime.create_or_load_user_session("SWAAG Orchestrator")
    config.tools.read_roots = [new]
    runtime._ensure_environment_initialized(state)
    assert state.environment.workspace.root == str(new)
    components = runtime._runtime_context_components(state, Mock())
    rendered = "\n".join(component.text for component in components)
    assert "Current filesystem workspace authority" in rendered
    assert "supersedes historical workspace errors" in rendered
    assert str(new) in rendered
