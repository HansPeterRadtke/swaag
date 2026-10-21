"""Source coverage is not a substitute for independently demonstrated effects."""
from __future__ import annotations
from hashlib import sha256
import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from infra_guideline_requirement_trace import generate, blocks
from infra_guideline_evidence_overlay import apply


def test_source_blocks_preserve_location_hash_and_entire_content():
    data = '# Top\n\nIntroduction line.\n\n## Rules\n\n- Must preserve user wording.\n  Even on next line.\n- Must check results.\n\n## Possibilities\n\n- Maybe use a graph.\n'
    rows = blocks(data, 'agents.md')
    assert [x['source_start_line'] for x in rows] == [3, 7, 9, 13]
    assert rows[1]['source_text'] == '- Must preserve user wording.\n  Even on next line.'
    assert rows[1]['section'] == ['Top', 'Rules']
    assert rows[1]['acceptance'] == 'NOT_VERIFIED'
    assert not rows[1]['proof']
    assert len(rows[1]['source_block_sha256']) == 64


def test_generate_records_every_source_document_and_preserved_original(tmp_path):
    guides = tmp_path/'guidlines'; guides.mkdir()
    (guides/'agents.md').write_text('# Agents\n\nDo not invent a result.\n')
    a = guides/'trading'; a.mkdir(); (a/'options.md').write_text('# Options\n\nKeep collateral.\n')
    originals = guides/'original-recordings'; originals.mkdir()
    raw = b'Original voice sentence exactly\n'; (originals/'first.txt').write_bytes(raw)
    output = tmp_path/'audit'
    result = generate(guides, output)
    assert result['documents'] == 2 and result['recordings'] == 1
    assert result['accepted_complete'] == 0
    trace = [json.loads(s) for s in (output/'requirement_source_trace.jsonl').read_text().splitlines()]
    assert len(trace) == result['source_blocks'] == 2
    assert all(x['acceptance'] == 'NOT_VERIFIED' and not x['proof'] for x in trace)
    manifest = json.loads((output/'source_manifest.json').read_text())
    assert manifest['original_recordings'][0]['sha256'] == sha256(raw).hexdigest()
    assert manifest['original_recordings'][0]['bytes'] == len(raw)


def test_overlay_requires_one_exact_source_and_never_certifies_whole_paragraph(tmp_path):
    guide = tmp_path/'guide'; guide.mkdir(); originals=guide/'original-recordings'; originals.mkdir()
    (originals/'one.txt').write_text('Original evidence')
    (guide/'agents.md').write_text('# Agent\n\nPreserve source. This is not proof that every rule works.\n')
    output = tmp_path/'trace'; generate(guide,output)
    records=[{'id':'source-preservation','verdict':'MECHANISM_TESTED_SCOPE_LIMITED',
              'owner':'SWAAG', 'source_file':'agents.md','source_phrase':'Preserve source.',
              'source_block_sha256':json.loads((output/'requirement_source_trace.jsonl').read_text().splitlines()[0])['source_block_sha256'],
              'implementation':'files/source.py','tests':'tests/test_source.py',
              'observation':'Test passed on isolated source copy.',
              'remaining':'All other source contexts are unverified.'}]
    src = tmp_path/'proof.json';src.write_text(json.dumps(records))
    result = apply(output/'requirement_source_trace.jsonl',src,tmp_path/'joined')
    assert result['reviewed_evidence_links']==1 and result['unverified_source_blocks']==0
    selected=json.loads((tmp_path/'joined'/'requirement_with_evidence.jsonl').read_text().strip())
    assert selected['acceptance']=='AT_LEAST_ONE_SUBCLAUSE_REVIEWED_NOT_WHOLE_BLOCK_CERTIFIED'
    assert selected['source_text']=='Preserve source. This is not proof that every rule works.'
    records[0]['source_phrase']='not in the file'
    src.write_text(json.dumps(records))
    with pytest.raises(ValueError,match='nonexistent'):
        apply(output/'requirement_source_trace.jsonl',src,tmp_path/'joined')


def test_overlay_rejects_duplicate_evidence_and_unearned_full_certification(tmp_path):
    guide=tmp_path/'guidlines';guide.mkdir()
    (guide/'agents.md').write_text('# Agent\n\nPreserve original user authority.\n')
    originals=guide/'original-recordings';originals.mkdir()
    (originals/'first.txt').write_text('Original user statement.')
    output=tmp_path/'index';generate(guide,output)
    template={
        'id':'task-authority','verdict':'MECHANISM_TESTED_SCOPE_LIMITED',
        'source_file':'agents.md','source_phrase':'Preserve original user authority.',
        'source_block_sha256':json.loads((output/'requirement_source_trace.jsonl').read_text().splitlines()[0])['source_block_sha256'],
        'owner':'SWAAG','implementation':'src/swaag/communication.py',
        'tests':'tests/test_control_authority.py',
        'observation':'One tested narrow receipt path.',
        'remaining':'Further model and real-world effects unverified.',
    }
    source=tmp_path/'checks.json'
    source.write_text(json.dumps([template,template]))
    with pytest.raises(ValueError,match='duplicate evidence ID'):
        apply(output/'requirement_source_trace.jsonl',source,tmp_path/'applied')
    duplicate={**template,'id':'second-review'}
    source.write_text(json.dumps([template,duplicate]))
    with pytest.raises(ValueError,match='duplicate source evidence claim'):
        apply(output/'requirement_source_trace.jsonl',source,tmp_path/'applied')
    unsupported={**template,'verdict':'FULLY_COMPLIANT'}
    source.write_text(json.dumps([unsupported]))
    with pytest.raises(ValueError,match='verdict'):
        apply(output/'requirement_source_trace.jsonl',source,tmp_path/'applied')


def test_review_evidence_requires_exact_unchanged_source_block(tmp_path):
    guides=tmp_path/'guidelines';guides.mkdir()
    original=guides/'original-recordings';original.mkdir()
    (original/'source.txt').write_text('Unmodified source recording')
    p=guides/'agents.md';p.write_text('# Agents\n\nPreserve the original user request.\n')
    index=tmp_path/'index';generate(guides,index)
    source=json.loads((index/'requirement_source_trace.jsonl').read_text().splitlines()[0])
    review={'id':'precise-proof','source_file':'agents.md',
            'source_phrase':'Preserve the original user request.',
            'source_block_sha256':source['source_block_sha256'],
            'verdict':'MECHANISM_TESTED_SCOPE_LIMITED','owner':'SWAAG',
            'implementation':'src/swaag/communication.py',
            'tests':'tests/test_control_authority.py',
            'observation':'Narrow source-level test passed.',
            'remaining':'No physical deployment acceptance.'}
    checks=tmp_path/'evidence.json';checks.write_text(json.dumps([review]))
    assert apply(index/'requirement_source_trace.jsonl',checks,tmp_path/'joined')['reviewed_evidence_links']==1
    p.write_text('# Agents\n\nPreserve the original user request. Additional rule added.\n')
    generate(guides,index)
    with pytest.raises(ValueError,match='source changed since evidence review'):
        apply(index/'requirement_source_trace.jsonl',checks,tmp_path/'joined')
