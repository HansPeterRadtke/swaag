from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace

import pytest

from swaag.background_work import BackgroundWork
from swaag.history import HistoryStore
from swaag.orchestration import OrchestrationManager, OrchestrationStore
from swaag.orchestration_api import OrchestrationApi
from swaag.tools.background_work import BackgroundWorkTool
from swaag.types import Message
@dataclass
class _Worker:
    worker_id: str
    status: str = "created"
    result: str | None = None
    error: str | None = None
    inference_weight: float = 1.0


class _Store:
    def __init__(self):
        self.items: dict[str, _Worker] = {}

    def get(self, worker_id: str) -> _Worker:
        if worker_id not in self.items:
            raise FileNotFoundError(worker_id)
        return self.items[worker_id]

    def set_inference_weight(self, worker_id: str, weight: float) -> _Worker:
        current = self.get(worker_id)
        updated = replace(current, inference_weight=float(weight))
        self.items[worker_id] = updated
        return updated


class _Workers:
    def __init__(self):
        self.store = _Store()
        self.created: list[str] = []
        self.canceled: list[str] = []

    def create(
        self,
        objective: str,
        *,
        name: str | None = None,
        inference_weight: float = 1.0,
    ):
        worker = _Worker(
            f"worker-{len(self.store.items)+1}",
            inference_weight=float(inference_weight),
        )
        self.store.items[worker.worker_id] = worker
        self.created.append(objective)
        return worker

    def start(self, worker_id: str):
        current = self.store.get(worker_id)
        updated = replace(current, status="queued")
        self.store.items[worker_id] = updated
        return updated


    def message(self, worker_id: str, message: str, *, source: str, resume_if_idle: bool = True):
        current = self.store.get(worker_id)
        return current

    def cancel(self, worker_id: str, *, reason: str):
        current = self.store.get(worker_id)
        updated = replace(current, status="canceled")
        self.store.items[worker_id] = updated
        self.canceled.append(worker_id)
        return updated


def setup(tmp_path, *, mode="authorized_backlog", maximum=128):
    workers=_Workers()
    manager=OrchestrationManager(workers,store=OrchestrationStore(tmp_path))
    backlog=BackgroundWork(manager,mode=mode,max_pending=maximum,foreground_busy=lambda:False)
    history=HistoryStore(tmp_path)
    state=history.create(config_fingerprint="test",model_base_url="http://test")
    authorization=history.record_event(state,"message_added",{"message":asdict(Message(
        role="user",content="After the current task finishes, run the queued repository checks.",created_at="now"))})
    return workers,manager,backlog,state,authorization


def plan(manager):
    item=manager.create_plan("Run authorized repository checks")
    manager.add_worker(item.plan_id,"Inspect the repository")
    return item.plan_id


def enqueue(backlog,plan_id,state,event):
    return backlog.enqueue(plan_id,authorization_session_id=state.session_id,authorization_event_sequence=event.sequence)


def test_default_finish_only_keeps_authorized_work_durable_without_starting(tmp_path):
    workers,manager,backlog,state,event=setup(tmp_path,mode="finish_only")
    plan_id=plan(manager)
    enqueue(backlog,plan_id,state,event)
    assert backlog.dispatch_once() is None
    assert manager.start_ready(plan_id)==[]  # normal polling cannot bypass idle policy
    assert workers.created==[]
    restarted=BackgroundWork(manager,foreground_busy=lambda:False)
    assert restarted.list()[0]["authorization_event_hash"]==event.hash
    assert restarted.dispatch_once() is None


def test_foreground_activity_and_active_plans_defer_dispatch(tmp_path):
    workers,manager,backlog,state,event=setup(tmp_path)
    plan_id=plan(manager)
    enqueue(backlog,plan_id,state,event)
    backlog.foreground_busy=lambda:True
    assert backlog.dispatch_once() is None
    backlog.foreground_busy=lambda:False
    foreground=plan(manager)
    manager.start_ready(foreground)
    assert backlog.dispatch_once() is None
    assert len(workers.created)==1
    manager.cancel_plan(foreground,reason="foreground complete for fixture")
    assert backlog.dispatch_once()==plan_id
    assert len(workers.created)==2
    assert backlog.dispatch_once() is None


def test_exact_retry_cancel_and_capacity_do_not_lose_queued_work(tmp_path):
    workers,manager,backlog,state,event=setup(tmp_path,maximum=1)
    first,second=plan(manager),plan(manager)
    original=enqueue(backlog,first,state,event)
    assert enqueue(backlog,first,state,event)==original
    with pytest.raises(ValueError,match="capacity"):
        enqueue(backlog,second,state,event)
    assert [item["plan_id"] for item in backlog.list()]==[first]
    assert backlog.cancel(first)
    assert not backlog.cancel(first)
    enqueue(backlog,second,state,event)
    assert backlog.dispatch_once()==second
    assert len(workers.created)==1


def test_plan_mutation_requires_renewed_authorization(tmp_path):
    workers,manager,backlog,state,event=setup(tmp_path)
    plan_id=plan(manager)
    enqueue(backlog,plan_id,state,event)
    manager.add_worker(plan_id,"Changed scope")
    assert backlog.dispatch_once() is None
    assert backlog.list()[0]["state"]=="held"
    assert workers.created==[]
    assert manager.start_ready(plan_id)==[]
    enqueue(backlog,plan_id,state,event)
    assert backlog.dispatch_once()==plan_id
    assert len(workers.created)==2


def test_missing_or_nonuser_authorization_is_rejected(tmp_path):
    workers,manager,backlog,state,event=setup(tmp_path)
    plan_id=plan(manager)
    history=HistoryStore(tmp_path)
    assistant=history.record_event(state,"message_added",{"message":asdict(Message(
        role="assistant",content="I invented a task",created_at="now"))})
    for sequence in (999999,assistant.sequence,True):
        with pytest.raises(ValueError):
            backlog.enqueue(plan_id,authorization_session_id=state.session_id,authorization_event_sequence=sequence)
    assert backlog.list()==[]
    assert workers.created==[]


def test_dispatch_commit_is_recovered_after_process_failure_without_second_plan(tmp_path,monkeypatch):
    workers,manager,backlog,state,event=setup(tmp_path)
    plan_id=plan(manager)
    enqueue(backlog,plan_id,state,event)
    advance=manager.advance
    def crash(_): raise RuntimeError("simulated process exit after dispatch commit")
    monkeypatch.setattr(manager,"advance",crash)
    with pytest.raises(RuntimeError,match="process exit"):
        backlog.dispatch_once()
    assert manager.store.get_plan(plan_id).status=="active"
    assert backlog.list()==[]
    monkeypatch.setattr(manager,"advance",advance)
    restarted=OrchestrationManager(workers,store=OrchestrationStore(tmp_path))
    restarted.advance_active_plans()
    restarted.advance_active_plans()
    assert len(workers.created)==1
    assert len(restarted.store.list_plans())==1


def test_concurrent_dispatchers_claim_only_one_plan(tmp_path):
    workers,manager,backlog,state,event=setup(tmp_path)
    ids=[plan(manager),plan(manager)]
    for plan_id in ids: enqueue(backlog,plan_id,state,event)
    other=BackgroundWork(manager,mode="authorized_backlog",foreground_busy=lambda:False)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(lambda queue:queue.dispatch_once(),[backlog,other]))
    assert sum(item is not None for item in results)==1
    assert len(workers.created)==1
    assert len(backlog.list())==1


def test_api_and_tool_are_explicit_about_mode_and_side_effects(tmp_path):
    _,manager,backlog,state,event=setup(tmp_path)
    api=OrchestrationApi(manager,background_work=backlog)
    plan_id=plan(manager)
    result=api.execute("backlog.enqueue",{"plan_id":plan_id,"authorization_session_id":state.session_id,"authorization_event_sequence":event.sequence})
    assert result["item"]["state"]=="pending"
    assert api.execute("backlog.list")["mode"]=="authorized_backlog"
    assert api.execute("backlog.cancel",{"plan_id":plan_id})["canceled"]
    tool=BackgroundWorkTool()
    assert tool.effective_kind(tool.validate({"operation":"list","plan_id":None,"authorization_event_sequence":None}))=="pure"
    assert tool.effective_kind(tool.validate({"operation":"enqueue","plan_id":plan_id,"authorization_event_sequence":event.sequence}))=="side_effect"
    assert set(tool.input_schema["required"])==set(tool.input_schema["properties"])


def test_autonomous_mode_declines_and_throttles_idea_generation(tmp_path, monkeypatch):
    workers,manager,_,_,_=setup(tmp_path)
    calls=[]
    backlog=BackgroundWork(
        manager, mode="autonomous_continue", foreground_busy=lambda:False,
        idea_generator=lambda: calls.append("called") or {
            "create": False, "objective": "", "worker_objective": "", "reason": "No useful task."},
        autonomous_idea_interval_seconds=60,
    )
    times=iter([100.0, 101.0])
    monkeypatch.setattr("swaag.background_work.time.monotonic", lambda: next(times))
    assert backlog.dispatch_once() is None
    assert backlog.dispatch_once() is None
    assert calls == ["called"]
    assert workers.created == []


def test_autonomous_mode_generates_one_auditable_plan(tmp_path, monkeypatch):
    workers,manager,_,_,_=setup(tmp_path)
    backlog=BackgroundWork(
        manager, mode="autonomous_continue", foreground_busy=lambda:False,
        idea_generator=lambda: {
            "create": True,
            "objective": "Improve repository verification",
            "worker_objective": "Inspect verification failures and propose one evidence-backed improvement",
            "reason": "Durable user history repeatedly prioritizes verification quality.",
        },
        autonomous_idea_interval_seconds=60,
    )
    monkeypatch.setattr("swaag.background_work.time.monotonic", lambda: 100.0)
    plan_id=backlog.dispatch_once()
    assert plan_id is not None
    snapshot=manager.store.snapshot(plan_id)
    assert snapshot["plan"].status == "active"
    assert workers.created == ["Inspect verification failures and propose one evidence-backed improvement"]
    generated=[event for event in snapshot["events"] if event["event_type"] == "autonomous_work_generated"]
    assert len(generated) == 1
    assert generated[0]["payload"]["reason"] == "Durable user history repeatedly prioritizes verification quality."
    assert backlog.dispatch_once() is None
    assert len(manager.store.list_plans()) == 1


def test_authorized_backlog_precedes_autonomous_idea_generation(tmp_path):
    workers,manager,_,state,event=setup(tmp_path)
    calls=[]
    backlog=BackgroundWork(
        manager, mode="autonomous_continue", foreground_busy=lambda:False,
        idea_generator=lambda: calls.append("called") or {
            "create": True, "objective": "invented", "worker_objective": "invented", "reason": "invented"},
    )
    plan_id=plan(manager)
    enqueue(backlog,plan_id,state,event)
    assert backlog.dispatch_once() == plan_id
    assert calls == []
    assert workers.created == ["Inspect the repository"]


def test_non_autonomous_modes_never_call_idea_generator(tmp_path):
    for mode in ("finish_only", "authorized_backlog"):
        workers,manager,_,_,_=setup(tmp_path)
        calls=[]
        backlog=BackgroundWork(
            manager, mode=mode, foreground_busy=lambda:False,
            idea_generator=lambda: calls.append("called") or {"create": False},
        )
        assert backlog.dispatch_once() is None
        assert calls == []
        assert workers.created == []
