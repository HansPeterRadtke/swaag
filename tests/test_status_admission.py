"""Independent status admission must not share the occupied planner pool."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import pytest
from swaag.communication import CommunicationService
from swaag.runtime import AgentRuntime
from swaag.capacity import QueueCapacityError

def test_status_dispatch_runs_while_planner_request_and_default_pool_are_full(make_config, monkeypatch):
    service=CommunicationService(AgentRuntime(make_config(),model_client=object()),max_concurrency=1)
    busy=threading.Event();release=threading.Event();status_entered=threading.Event()
    def blocking():
        busy.set()
        assert release.wait(10)
    def answer(session,question):
        status_entered.set()
        return 'The planner is still working.'
    monkeypatch.setattr(service,'answer_status_question',answer)
    async def exercise():
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=1))
        async with service._semaphore:
            occupied=asyncio.create_task(asyncio.to_thread(blocking))
            while not busy.is_set():await asyncio.sleep(.01)
            status=asyncio.create_task(service._dispatch_json_line_request({'op':'ask_status','session':'target','question':'What is happening?'}))
            try:
                answer_result=await asyncio.wait_for(asyncio.shield(status),2)
                assert answer_result=={'answer':'The planner is still working.'}
                assert not occupied.done()
            finally:
                release.set()
                await occupied
                if not status.done():status.cancel()
                await asyncio.gather(status,return_exceptions=True)
    try:asyncio.run(exercise())
    finally:release.set();service.close()

def test_canceled_status_transport_does_not_release_occupied_capacity(make_config,monkeypatch):
    config=make_config(communication__max_pending_requests=1)
    service=CommunicationService(AgentRuntime(config,model_client=object()),max_concurrency=1)
    entered=threading.Event();release=threading.Event()
    def answer(session,question):
        entered.set()
        assert release.wait(10)
        return 'Current status.'
    monkeypatch.setattr(service,'answer_status_question',answer)
    async def exercise():
        request=asyncio.create_task(service._dispatch_json_line_request({'op':'ask_status','question':'Status?'}))
        while not entered.is_set():await asyncio.sleep(.01)
        request.cancel()
        with pytest.raises(asyncio.CancelledError):await request
        try:
            with pytest.raises(QueueCapacityError,match='status.*capacity'):
                await service._dispatch_json_line_request({'op':'ask_status','question':'Again?'})
        finally:release.set()
        deadline=asyncio.get_running_loop().time()+3
        while asyncio.get_running_loop().time()<deadline:
            try:
                result=await service._dispatch_json_line_request({'op':'ask_status','question':'Now?'})
                assert result=={'answer':'Current status.'}
                return
            except QueueCapacityError:await asyncio.sleep(.01)
        pytest.fail('Finished status did not release admission capacity')
    try:asyncio.run(exercise())
    finally:release.set();service.close()



def test_status_backend_failure_releases_admission(make_config,monkeypatch):
    service=CommunicationService(AgentRuntime(make_config(communication__max_pending_requests=1),model_client=object()),max_concurrency=1)
    calls=[]
    def answer(*args):
        calls.append(args)
        if len(calls)==1:raise RuntimeError('backend failed')
        return 'Recovered status.'
    monkeypatch.setattr(service,'answer_status_question',answer)
    async def exercise():
        with pytest.raises(RuntimeError,match='backend failed'):
            await service._dispatch_json_line_request({'op':'ask_status'})
        assert await service._dispatch_json_line_request({'op':'ask_status'})=={'answer':'Recovered status.'}
    try:asyncio.run(exercise())
    finally:service.close()


def test_status_executor_preserves_request_context(make_config, monkeypatch):
    from contextvars import ContextVar
    correlation = ContextVar('status_request_correlation', default='missing')
    service = CommunicationService(
        AgentRuntime(make_config(), model_client=object()), max_concurrency=1
    )
    monkeypatch.setattr(service, 'answer_status_question', lambda *_: correlation.get())
    async def exercise():
        token = correlation.set('request-trace')
        try:
            result = await service._dispatch_json_line_request({'op': 'ask_status'})
            assert result == {'answer': 'request-trace'}
        finally:
            correlation.reset(token)
    try:
        asyncio.run(exercise())
    finally:
        service.close()
