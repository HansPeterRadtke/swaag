import pytest

from swaag.grammar import yes_no_contract
from swaag.prompt_instructions import make_prompt_instruction
from swaag.runtime import AgentRuntime
from swaag.tokens import ExactTokenCounter
from swaag.types import PromptComponent


class NextProjectionCaptured(Exception):
    pass


def test_nonfitting_projection_retry_changes_effective_model_target(make_config, monkeypatch):
    config = make_config(model__context_limit=1500,
                         prompt_instructions__max_instruction_chars=60000,
                         prompt_instructions__max_total_chars=64000)
    runtime = AgentRuntime(config, model_client=object(), token_counter=ExactTokenCounter(
        lambda text: len(text.split()) if text.strip() else 0))
    state = runtime.create_or_load_session()
    state.prompt_instructions.append(make_prompt_instruction(
        config, title='Exact operating policy', content='Required policy. ' * 2000, scopes=['action']))
    assembly = runtime.prompts.build_semantic_operation_prompt(
        kind='action', system_instruction='Follow the user request.',
        components=[PromptComponent(name='request', category='current_user', text='Continue.')])
    failed = runtime._compile_context(state, assembly, yes_no_contract(), minimum_output_tokens=64,
                                     context_limit_resolution=(1500, 'test'))
    assert not failed.report.fits
    compile_context = runtime._compile_context
    def compile_nonfitting_projection(state, candidate, *args, **kwargs):
        compiled = compile_context(state, candidate, *args, **kwargs)
        if 'REPEATED_PROJECTION' in candidate.prompt_text:
            compiled.report.fits = False
            compiled.report.required_tokens = compiled.report.context_limit + 1
        return compiled
    monkeypatch.setattr(runtime, '_compile_context', compile_nonfitting_projection)
    targets = []
    def reduce(*args, **kwargs):
        targets.append(kwargs['target_tokens'])
        if len(targets) == 2:
            raise NextProjectionCaptured()
        # The inner reducer had to cap the actual model target below the outer
        # requested target. A retry must change that effective request.
        kwargs['build_assembly'](kwargs['source_text'], kwargs['source_label'], 64)
        return 'REPEATED_PROJECTION', failed.report
    monkeypatch.setattr(runtime, '_reduce_text_hierarchically', reduce)
    with pytest.raises(NextProjectionCaptured):
        runtime._recover_system_prompt_contributor_overflow(
            state, assembly, yes_no_contract(), failed, minimum_output_tokens=64,
            context_limit_resolution=(1500, 'test'))
    assert targets[0] > 64
    assert 32 <= targets[1] < 64
