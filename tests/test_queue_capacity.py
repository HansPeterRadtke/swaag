from concurrent.futures import Future, ThreadPoolExecutor
import time

import pytest

from swaag.capacity import QueueCapacityError, positive_capacity
from swaag.communication import CommunicationStore
from swaag.history import HistoryStore
from swaag.inference import InferenceRequestCoordinator
from swaag.runtime import AgentRuntime
from swaag.workers import WorkerManager, WorkerStore


def test_worker_admission_is_atomic_across_managers_and_rejects_without_mutation(tmp_path):
    stores=[WorkerStore(tmp_path,max_active_workers=1),WorkerStore(tmp_path,max_active_workers=1)]
    workers=[stores[0].create("session-one","one"),stores[0].create("session-two","two")]
    def admit(index):
        try:
            stores[index].transition(workers[index].worker_id,"queued",expected={"created"})
            return True
        except QueueCapacityError:
            return False
    with ThreadPoolExecutor(max_workers=2) as pool:
        accepted=list(pool.map(admit,range(2)))
    assert sum(accepted)==1
    winner=accepted.index(True);loser=1-winner
    assert stores[0].get(workers[loser].worker_id).status=="created"
    stores[0].transition(workers[winner].worker_id,"completed")
    assert admit(loser)
    assert stores[0].get(workers[winner].worker_id).status=="completed"


def test_control_pressure_preserves_exact_retry_and_rejects_identifier_rebinding(tmp_path):
    store=HistoryStore(tmp_path,max_pending_controls=1)
    first=store.enqueue_control_message("session-one","continue",control_id="control_one")
    assert store.enqueue_control_message("session-one","continue",control_id="control_one")==first
    with pytest.raises(QueueCapacityError):
        store.enqueue_control_message("session-one","another",control_id="control_two")
    with pytest.raises(ValueError,match="already bound"):
        store.enqueue_control_message("session-two","different",control_id="control_one")
    assert [item["message"] for item in store.list_pending_control_messages("session-one")]==["continue"]
    store.mark_control_message_processed("session-one","control_one")
    assert store.enqueue_control_message("session-one","another",control_id="control_two")["message"]=="another"
    assert len(store.list_authoritative_control_messages("session-one"))==2


def test_communication_capacity_includes_processing_and_retains_terminal_evidence(tmp_path):
    store=CommunicationStore(tmp_path,max_pending_requests=1)
    first=store.create("session","first")
    store.set_status(first.correlation_id,"processing")
    with pytest.raises(QueueCapacityError): store.create("session","second")
    store.set_status(first.correlation_id,"completed",reply="done")
    second=store.create("session","second")
    assert store.next_pending().correlation_id==second.correlation_id
    assert store.get(first.correlation_id).reply=="done"


def test_inference_capacity_is_per_backend_and_counts_suspended_replays(tmp_path):
    def coordinator(key):
        return InferenceRequestCoordinator(tmp_path,backend_key=key,capacity_resolver=lambda:(1,"test"),max_pending_requests=1)
    def enqueue(coordinator,call):
        return coordinator.enqueue(session_id="s",run_id="r",call_id=call,call_kind="agent_action",priority=0,source="test")
    first=coordinator("first");second=coordinator("second")
    request=enqueue(first,"call-one")
    first.acquire(request.request_id)
    first.suspend(request.request_id,reason="temporary control preemption")
    with pytest.raises(QueueCapacityError): enqueue(first,"call-two")
    assert enqueue(second,"call-three").status=="queued"
    first.resume(request.request_id)
    first.acquire(request.request_id)
    first.complete(request.request_id)
    assert enqueue(first,"call-four").status=="queued"
    assert first.get(request.request_id).status=="completed"


def test_completed_worker_futures_are_released_and_later_work_still_runs(make_config):
    config=make_config(communication__max_active_workers=2)
    manager=WorkerManager(AgentRuntime(config,model_client=object()),max_workers=1)
    manager._run_worker_with_context=lambda worker_id:manager.store.transition(worker_id,"completed",result="done")
    try:
        for index in range(30):
            worker=manager.create(f"small job {index}")
            manager.start(worker.worker_id)
            manager.wait(worker.worker_id,timeout_seconds=3)
        manager.shutdown()
        assert manager._futures=={}
        assert manager._deferred_contexts=={}
        assert len(manager.store.list(statuses={"completed"}))==30
    finally:
        manager.shutdown()


def test_deferred_callback_does_not_duplicate_an_already_replaced_future(make_config):
    manager=WorkerManager(AgentRuntime(make_config(),model_client=object()))
    worker=manager.create("queued work")
    manager.store.transition(worker.worker_id,"queued")
    replacement=Future()
    manager._futures[worker.worker_id]=replacement
    manager._deferred_submissions.add(worker.worker_id)
    try:
        manager._submit_deferred(worker.worker_id)
        assert manager._futures[worker.worker_id] is replacement
        assert not replacement.done()
    finally:
        replacement.cancel()
        manager.shutdown()


@pytest.mark.parametrize("value",[0,-1,True,1.5,float("nan"),"3"])
def test_capacity_contract_rejects_invalid_values(value):
    with pytest.raises(ValueError): positive_capacity(value,"test")


def test_token_count_memo_is_bounded_and_evicted_values_are_recomputed(make_config):
    from types import SimpleNamespace
    config=make_config(context__max_token_count_cache_entries=2)
    calls=[]
    def tokenize(text):
        calls.append(text)
        return len(text)
    runtime=AgentRuntime(config,model_client=SimpleNamespace(tokenize=tokenize))
    state=runtime.create_or_load_session()
    for text in ('alpha','beta','gamma','alpha'):
        assert runtime._tokenize_with_history(state,text).tokens==len(text)
        assert len(runtime._token_count_cache)<=2
    assert calls==['alpha','beta','gamma','alpha']
    assert len([event for event in runtime.history.read_history(state.session_id)
                if event.event_type=='model_tokenize_result'])==4
