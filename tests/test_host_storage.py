"""Read-only host headroom must be precise and present in the fast context."""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from swaag.host_storage import capture_host_storage
from tests.test_fast_foreground_reply import setup, fast
from tests.test_fast_worker_state import snapshot
from swaag.foreground_fast import generate_fast_foreground_response


def test_storage_uses_nonroot_available_space_and_no_optimistic_free_percentage(monkeypatch):
    import swaag.host_storage as storage
    monkeypatch.setattr(storage.os,'statvfs',lambda path:SimpleNamespace(
        f_frsize=1024**3, f_blocks=1000, f_bfree=150, f_bavail=100,
    ))
    result=capture_host_storage('/fake/swaag/sessions')
    assert result['observation_available'] is True
    assert result['available_gib']==100.0
    assert result['filesystem_total_gib']==1000.0
    # It is not accurate to call the 50 reserved GiB available to SWAAG.
    assert result['used_percent_of_user_accessible_capacity']==89.5
    assert 'shared with other projects' in result['scope']
    assert 'not project-specific' in result['evidence']


def test_storage_failure_is_qualified_and_never_returns_an_invented_value(monkeypatch):
    import swaag.host_storage as storage
    def unavailable(path):
        raise OSError('Unable to inspect filesystem')
    monkeypatch.setattr(storage.os,'statvfs',unavailable)
    result=capture_host_storage('/unavailable')
    assert result['observation_available'] is False
    assert 'available_gib' not in result
    assert 'used_percent_of_user_accessible_capacity' not in result


def test_invalid_kernel_counters_are_not_published(monkeypatch):
    import swaag.host_storage as storage
    monkeypatch.setattr(storage.os,'statvfs',lambda path:SimpleNamespace(
        f_frsize=0, f_blocks=100, f_bfree=1, f_bavail=-1,
    ))
    assert capture_host_storage('/fake')['observation_available'] is False


def test_live_style_storage_facts_reach_one_fast_coder_context(make_config):
    service,runtime,client=setup(make_config,[fast('answer','About twenty-five GiB is available.')])
    try:
        state=service._orchestrator_state()
        data=snapshot(with_progress=False)
        data['captured_at']=datetime.now(timezone.utc).isoformat()
        data['host_storage']={
            'observation_available':True,'available_gib':25.0,
            'used_percent_of_user_accessible_capacity':98.0,
            'scope':'filesystem containing the SWAAG session store; shared with other projects',
            'observed_at':datetime.now(timezone.utc).isoformat(),
        }
        result=generate_fast_foreground_response(runtime,state,
            'How much free disk space does Thor have?',observed_world=data)
        assert result and result.mode=='answer'
        assert len(client.requests)==1
        assert client.requests[0]['contract']=='fast_foreground_response'
        prompt=client.requests[0]['prompt']
        assert 'available_gib' in prompt and '25.0' in prompt
        assert 'shared with other projects' in prompt
        assert '98.0' in prompt
    finally:service.close()


def test_actual_background_world_snapshot_captures_storage_without_mutations(make_config,monkeypatch):
    service,runtime,client=setup(make_config,[fast('answer','There is available disk space.')])
    try:
        import swaag.host_storage as storage
        monkeypatch.setattr(storage,'capture_host_storage',lambda root:{
            'observation_available':True,'available_gib':18.0,
            'used_percent_of_user_accessible_capacity':97.0,
            'scope':'the SWAAG sessions filesystem',
        })
        service._refresh_orchestrator_snapshot()
        state=service._fast_foreground_state
        assert isinstance(state,dict)
        assert state['host_storage']['available_gib']==18.0
        assert not any(e.event_type=='tool_called' for e in
                       runtime.history.read_history(service._orchestrator_state().session_id))
    finally:service.close()
