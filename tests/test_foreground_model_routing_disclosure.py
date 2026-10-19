"""Model identity must come from the actual foreground runtime, not model guesses."""

from swaag.communication import CommunicationService
from swaag.runtime import AgentRuntime
from swaag.snapshot_view import snapshot_prompt_view


def test_foreground_snapshot_distinguishes_actual_configured_coder_and_default_worker(make_config):
    config = make_config(
        model__base_url='http://127.0.0.1:18735/v1',
        model__model_identity='fixture-q8-worker-model',
        model__profile_name='fixture-q8-worker-profile',
        communication__model_base_url='http://127.0.0.1:18733/v1',
        communication__model_identity='fixture-coder-foreground-model',
        communication__model_profile_name='fixture-coder-foreground-profile',
    )
    service=CommunicationService.from_runtime(AgentRuntime(config))
    try:
        snapshot=service._orchestrator_runtime_snapshot()
        routing=snapshot['model_routing']
        assert routing['foreground']['requested_model_id']=='fixture-coder-foreground-model'
        assert routing['foreground']['configured_profile']=='fixture-coder-foreground-profile'
        assert routing['foreground']['base_url']=='http://127.0.0.1:18733/v1'
        assert routing['default_worker']['requested_model_id']=='fixture-q8-worker-model'
        assert routing['default_worker']['base_url']=='http://127.0.0.1:18735/v1'
        assert routing['evidence_scope']=='configured SWAAG runtime routes, not verified remote model binaries'
        exposed=snapshot_prompt_view(snapshot,123)
        assert exposed['model_routing']==routing
        # Do not confuse the worker/model that performs delegated tasks with
        # the current foreground response author or invent a provider identity.
        assert routing['foreground']['requested_model_id']!=routing['default_worker']['requested_model_id']
        # Displaying a configured model's ID is not authorization to use that
        # string as a plan-node model_key. The exact named routing table owns
        # those keys; Q8 is already the default worker model.
        assert routing['available_worker_model_keys'] == ['default']
        assert "default" in routing['worker_model_key_rule']
        assert "requested_model_id" in routing['worker_model_key_rule']
        assert 'fixture-q8-worker-model' not in routing['available_worker_model_keys']
    finally:
        service.close()


def test_named_worker_routes_are_exposed_as_keys_not_model_ids(make_config):
    """Named worker routing remains configurable and is never an alias lookup."""
    config = make_config(
        model__model_identity='resident-q8-default-model',
        communication__model_identity='resident-coder-foreground-model',
        communication__enabled=True,
        communication__model_routes={
            'background_alt': 'http://127.0.0.1:18925/v1',
            'another': 'http://127.0.0.1:18926/v1',
        },
    )
    service = CommunicationService.from_runtime(AgentRuntime(config))
    try:
        routing = service._orchestrator_runtime_snapshot()['model_routing']
        assert routing['available_worker_model_keys'] == [
            'default', 'another', 'background_alt']
        assert 'resident-q8-default-model' not in routing['available_worker_model_keys']
        assert 'resident-coder-foreground-model' not in routing['available_worker_model_keys']
        assert 'null' in routing['worker_model_key_rule']
        assert 'requested_model_id' in routing['worker_model_key_rule']
    finally:
        service.close()
