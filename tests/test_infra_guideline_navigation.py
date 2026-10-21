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
