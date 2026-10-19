"""Fast user-facing tool discovery keeps every available name but no raw schema."""
from swaag.prompts import PromptBuilder
from swaag.tools.registry import ToolRegistry


def test_compact_index_retains_every_capability_without_exposing_long_tool_schema(make_config):
    cfg=make_config()
    cfg.tools.enabled = ['orchestration_control','runtime_snapshot_lookup','agent_control',
                         'questions','worker_questions','list_files','calculator']
    tool_registry=ToolRegistry()
    full=[]
    for name in cfg.tools.enabled:
        tool=tool_registry.get(name)
        full.append((name,tool.description,tool.usage_guidance))
    builder=PromptBuilder(cfg)
    short=builder.render_capability_index(full, compact=True)
    verbose=builder.render_capability_index(full, compact=False)
    assert set(line.split(':',1)[0].removeprefix('- ') for line in short.splitlines()) == set(cfg.tools.enabled)
    assert 'orchestration_control' in short and 'Plan, start, control' in short
    assert 'agent_control' in short and 'durable instruction' in short
    assert 'runtime_snapshot_lookup' in short and 'snapshot' in short
    assert len(short)<len(verbose)
    assert 'input_schema' not in short and 'usage_guidance:' not in short
    assert 'plan_spec' not in short


def test_unrecognized_connected_capability_keeps_full_description():
    from swaag.prompts import PromptBuilder
    assert 'Carefully inspect imported records' in PromptBuilder.__new__(PromptBuilder).render_capability_index(
        [('custom_provider_tool', 'Carefully inspect imported records', '')], compact=True)
