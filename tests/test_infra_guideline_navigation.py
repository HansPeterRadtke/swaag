"""Project-specific owner guidelines must be discoverable by the full agent."""
from pathlib import Path
from swaag.system_context import runtime_system_context_sources
from tests.test_tools import _empty_state


def test_trusted_readable_guideline_index_is_exposed_without_loading_whole_tree(tmp_path, make_config, monkeypatch):
    import swaag.system_context as context
    guide = tmp_path/'infra'/'docs'/'guidlines'
    guide.mkdir(parents=True)
    (guide/'README.md').write_text('# Guide index\nMandatory user goals and input authority.\n')
    (guide/'agents.md').write_text('# Agent rules\n' + 'large content ' * 1000)
    (guide/'trading').mkdir()
    (guide/'trading'/'data_and_backtesting.md').write_text('Keep source vintages.')
    monkeypatch.setattr(context,'INFRA_GUIDELINES_ROOT',guide)
    cfg=make_config(tools__enabled=['read_file'],tools__read_roots=[tmp_path])
    found=runtime_system_context_sources(cfg,_empty_state())
    matches=[x for x in found if x.name=='infra_guideline_navigation']
    assert len(matches)==1
    entry=matches[0]
    assert str(guide/'README.md') in entry.render()
    assert 'agents.md' in entry.render() and 'trading' in entry.render()
    assert 'owner-provided policy' in entry.introduction
    assert 'security/privacy' in entry.render()
    assert 'recording conversion' in entry.render()
    assert 'Keep source vintages.' not in entry.render()
    assert 'large content' not in entry.render()
    assert len(entry.render())<2400
    assert entry.optional is False
    assert entry.projectable is False


def test_guidelines_not_announced_without_read_access(tmp_path, make_config, monkeypatch):
    import swaag.system_context as context
    guide=tmp_path/'infra'/'docs'/'guidlines';guide.mkdir(parents=True)
    (guide/'README.md').write_text('Index')
    monkeypatch.setattr(context,'INFRA_GUIDELINES_ROOT',guide)
    for config in (
        make_config(tools__enabled=['read_file'],tools__read_roots=[tmp_path/'other']),
        make_config(tools__enabled=['echo'],tools__read_roots=[tmp_path]),
    ):
        assert not any(x.name=='infra_guideline_navigation'
                       for x in runtime_system_context_sources(config,_empty_state()))


def test_guidelines_do_not_follow_unapproved_root_alias(tmp_path, make_config, monkeypatch):
    import swaag.system_context as context
    other=tmp_path/'private';other.mkdir();(other/'README.md').write_text('secret')
    link=tmp_path/'allowed'/'guidlines';link.parent.mkdir();link.symlink_to(other,target_is_directory=True)
    monkeypatch.setattr(context,'INFRA_GUIDELINES_ROOT',link)
    cfg=make_config(tools__enabled=['read_file'],tools__read_roots=[tmp_path/'allowed'])
    assert not any(x.name=='infra_guideline_navigation'
                   for x in runtime_system_context_sources(cfg,_empty_state()))


def test_infra_guideline_navigation_mentions_owner_rules_and_scoped_text_guides(tmp_path, make_config, monkeypatch):
    import swaag.system_context as context
    infra=tmp_path/'infra'
    guides=infra/'docs'/'guidlines';guides.mkdir(parents=True)
    (guides/'README.md').write_text('# Guide root\n')
    (guides/'agents.md').write_text('Use original user message.\n')
    (guides/'tests.txt').write_text('Old supplementary testing procedures; see testing.md.\n')
    (guides/'constraint_decoding.txt').write_text('Structured output provider-specific schema.\n')
    (infra/'AGENTS.md').write_text('Root Infra ownership instructions.\n')
    (infra/'README.md').write_text('Root Infra component index.\n')
    monkeypatch.setattr(context,'INFRA_GUIDELINES_ROOT',guides)
    config=make_config(tools__enabled=['read_file'],tools__read_roots=[infra])
    source=next(x for x in runtime_system_context_sources(config,_empty_state())
                if x.name=='infra_guideline_navigation')
    assert str(infra/'AGENTS.md') in source.render()
    assert str(infra/'README.md') in source.render()
    assert str(guides/'tests.txt') in source.render()
    assert str(guides/'constraint_decoding.txt') in source.render()
    assert 'only for Infra-owned operations' in source.render()
    assert 'testing.md takes precedence' in source.render()
    assert 'OpenRouter' in source.render()
    assert 'Root Infra ownership instructions.' not in source.render()


def test_infra_navigation_does_not_expose_owner_rules_outside_capability(tmp_path, make_config, monkeypatch):
    import swaag.system_context as context
    infra=tmp_path/'infra';guides=infra/'docs'/'guidlines';guides.mkdir(parents=True)
    (guides/'README.md').write_text('Index')
    (infra/'AGENTS.md').write_text('Repository owner rules')
    (infra/'README.md').write_text('Infra repository overview')
    monkeypatch.setattr(context,'INFRA_GUIDELINES_ROOT',guides)
    config=make_config(tools__enabled=['read_file'],tools__read_roots=[guides])
    source=next(x for x in runtime_system_context_sources(config,_empty_state())
                if x.name=='infra_guideline_navigation')
    assert str(infra/'AGENTS.md') not in source.render()
    assert 'infra_repo_scope_rules' not in source.render()
    assert 'infra_repo_overview' not in source.render()


def test_infra_root_rules_symlink_cannot_escape_configured_read_scope(tmp_path, make_config, monkeypatch):
    import swaag.system_context as context
    infra = tmp_path/'infra'
    guides = infra/'docs'/'guidlines'
    guides.mkdir(parents=True)
    (guides/'README.md').write_text('# Guide index')
    hidden = tmp_path/'unapproved'/'AGENTS.md'
    hidden.parent.mkdir()
    hidden.write_text('Private rule outside authorized source root')
    (infra/'AGENTS.md').symlink_to(hidden)
    monkeypatch.setattr(context, 'INFRA_GUIDELINES_ROOT', guides)
    config = make_config(tools__enabled=['read_file'], tools__read_roots=[infra])
    source = next(item for item in runtime_system_context_sources(config, _empty_state())
                  if item.name == 'infra_guideline_navigation')
    assert 'infra_repo_scope_rules' not in source.render()
    assert str(hidden) not in source.render()
    assert 'Private rule' not in source.render()
