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
    finally:
        service.close()
