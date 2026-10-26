#!/usr/bin/env python3
"""Publish compact, reproducible end-to-end actual SWAAG attachment receipts."""
from pathlib import Path
import hashlib
import json
import sqlite3
import statistics

ROOT=Path('/data/var/perceptual-independent-20261024/swaag_real_android_tool_roundtrip_20261027')
SQLITE=Path('/data/var/perceptual-independent-20261024/sqlite-test-runtime-20261027')
REPO=Path(__file__).resolve().parents[2]
DEST=REPO/'docs/research/perceptual-real-swaag-attachment-snapshot-roundtrip-2026-10-27.json'
DOC=DEST.with_suffix('.md')


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def main():
    if DEST.exists() or DOC.exists():raise RuntimeError('Previously published report exists')
    frozen=ROOT/'frozen_manifest.json';executed=ROOT/'execution.json';recovered=ROOT/'recovery.json'
    protocol=json.loads(frozen.read_text());data=json.loads(executed.read_text());restore=json.loads(recovered.read_text())
    assert data['frozen_manifest_sha']==sha(frozen)
    assert restore['frozen_manifest_SHA']==sha(frozen)
    assert restore['execution_SHA']==sha(executed)
    assert set(protocol['derived_sources'])==set(data['cases'])==set(restore['cases'])=={'clean','changed','wrong_activity'}
    assert data['sqlite_version']=='3.54.1'
    assert restore['trusted_host_capability_was_volatile_not_in_journal']
    assert restore['all_archived_session_journals_had_exact_lineage']
    outcomes={}
    for name,first in data['cases'].items():
        rec=restore['cases'][name];result=first['tool_output']
        assert result['verdict']==protocol['expected'][name]
        assert rec['rebound_tool_verdict']==result['verdict']
        assert rec['result_events_persisted']==2
        assert rec['no_capability_failure_is_explicit'] and rec['archived_safely']
        assert rec['reentry_still_has_source_attachments']
        assert result['property_id']=='exact_immutable_png_snapshot_bytes'
        assert result['semantic_visual_quality_evaluated'] is False
        assert result['universal_media_release_approved'] is False
        assert len(result['source_event_references'])==2
        assert all(x['hash'] for x in result['source_event_references'])
        assert first['source_approved_sha']==protocol['derived_sources'][name]['approved']['sha256']
        assert first['actual_sha']==protocol['derived_sources'][name]['actual']['sha256']
        outcomes[name]={
            'original_author_output_source_state_sha256':first['source_approved_sha'],
            'actual_user_visible_PNG_source_sha256':first['actual_sha'],
            'trusted_original_manifest_content_SHA':result['host_manifest_sha256'],
            'source_original_state_from_pinned_author':protocol['derived_sources'][name]['approved']['state']['state_id'],
            'target_original_state_from_pinned_author':protocol['derived_sources'][name]['actual']['state']['state_id'],
            'source_image_native_size_wh':protocol['derived_sources'][name]['approved']['original_dimensions'],
            'target_image_native_size_wh':protocol['derived_sources'][name]['actual']['original_dimensions'],
            'first_real_runtime_tool_verdict':result['verdict'],
            'restarted_runtime_verdict':rec['rebound_tool_verdict'],
            'source_attestor_missing_after_restart_refused':rec['no_capability_failure_is_explicit'],
            'persisted_result_event_count':rec['result_events_persisted'],
            'original_first_tool_result_event_SHA256':first['tool_result_event_hash'],
            'rebound_tool_result_event_SHA256':rec['rebind_result_event_hash'],
            'saved_journal_archive_event_count':rec['archive_event_count'],
            'archive_reentry_kept_attachments':rec['reentry_still_has_source_attachments'],
            'raw_png_decode_wall_seconds':first['raw_source_PNGs_decoded_as_real_images_s'],
            'original_tool_dispatch_wall_s':first['duration_tool_once_s'],
            'session_id_SHA256':hashlib.sha256(first['session_id'].encode()).hexdigest(),
            'not_general_Graphics_GUI_visual_quality_verdict':True,
        }
    assert outcomes['clean']['first_real_runtime_tool_verdict']=='pass'
    assert outcomes['changed']['first_real_runtime_tool_verdict']=='fail'
    assert outcomes['wrong_activity']['first_real_runtime_tool_verdict']=='review'
    tool_times=[x['original_tool_dispatch_wall_s'] for x in outcomes.values()]
    source=SQLITE/'sqlite-amalgamation-3540100/sqlite3.c'
    assert source.is_file()
    assert sqlite3.sqlite_version=='3.54.1'
    result={
        'date':'2026-10-27','status':'PARTIAL',
        'actual_research_goal':'Complete real source-to-target tool workflow for original captured Android PNGs through SWAAG attachment persistence, direct tool dispatch, saved history, fresh process and archive/reentry; not a live production GUI acceptance',
        'original_screenshot_source_release_SHA256':protocol['input_source_sha256'],
        'pinned_source_manifest_SHA256':sha(frozen),
        'first_runtime_execution_SHA256':sha(executed),
        'separate_restarted_recovery_SHA256':sha(recovered),
        'new_registered_tool':'verify_immutable_snapshot',
        'new_opt_in_host_only_capability':'immutable_snapshot_attestor',
        'tool_registered_in_real_SWAAG_ToolRegistry':True,
        'tool_default_disabled_and_host_capability_required':True,
        'verified_host_manifest_forgery_limit':'The research harness reconstructs public source identity from a pinned publisher file manifest, not a live Android Activity or authenticated runtime screen capture. Host capability is in-memory and must be trusted/rebound; the tool does not certify any untrusted caller flags.',
        'real_source_to_target_end_state_results':outcomes,
        'durable_storage':'Three entirely isolated test sessions in SWAAG SQLite+hash-chain journals, two attachment_added sources per session, tool_result persisted before restart, explicit missing-capability refusal on new runtime, separate host source rebind, second real result, archived journals and actual restored source attachment references',
        'model_inference_or_gpu_call':False,
        'resource':{
          'runtime_tool_s_min':min(tool_times),
          'runtime_tool_s_max':max(tool_times),
          'runtime_tool_s_median':round(statistics.median(tool_times),4),
          'input_original_images_decoder':'actual ffmpeg PNG decoder, each genuine public Android PNG was decoded (not just header/filename)',
          'all_initial_calls_wall_s':data['total_wall_s'],
          'separate_process_recovery_wall_s':restore['CPU_wall_s'],
          'sqlite_runtime_version':data['sqlite_version'],
          'isolated_SQLite_amalgamation_3540100_SHA256':sha(source),
          'local_SWAAG_runtime_venv_not_system_or_production':True,
          'tested_under_Nitro_contention_not_a_general_latency_guarantee':True,
          'active_GPU_models_unchanged':True,
        },
        'negative_and_boundary_controls':{
          'wrong_activity_same_app_original_author_source_1080p':'REVIEW, not automatic visual FAIL',
          'missing_capability_after_fresh_process':'No tool_result, explicit runtime error; no impersonation of saved attestation',
          'source_rev_and_artifact_hash_mismatches':'REVIEW in six new unit tests; no arbitrary model-supplied attestation',
          'new_tool_only_accepts_originalPNG_mime_and_matching_attested_IHDR_dimensions':'True',
          'source_actor_source_revision_and_native_end_dimensions_checked':'True',
          'session_archive_reentry_verified':'True',
        },
        'scope_and_unproved':[
          'Genuine captured Android GUI PNGs, but source-author mutations rather than a live Android device render. No independent human re-rating of clipped content.',
          'Attestation is trusted because a private RESEARCH host harness pins publisher metadata; no authenticated, continuously observed Android Activity has been integrated.',
          'The production default registered tool list does not enable this opt-in tool; no production SWAAG worker or LLM prompt-driven call was invoked.',
          'Exact byte equality is only a valid contract for immutable screenshots, not dynamic UI, contrast, semantic overlaps, arrowhead directions or human visual acceptance.',
          'Final 3D photorealism, naturally occurring temporal faults, true audio/video alignment and source-certified mechanical part completeness remain unqualified.',
          'One earlier storyboard transfer was denied and not retried; no production assets, active sessions or shared GPU services were changed.',
        ],
    }
    DEST.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    DOC.write_text(f'''# SWAAG real attachment-to-verdict-to-saved-session round-trip — October 27, 2026

**Result: NARROW END-TO-END TEST PASS in a separate, actual SWAAG runtime process. Overall perception goal PARTIAL.** This round rechecked the full original owner testing recording, Nitro's live `/data/infra/AGENTS.md`, GUI guidelines, current branch and the previous independent real Android/3D/video false-positive data before editing. The research worktree was clean at commit `1b22dbd`. No production service, media or old denied external transfer was touched.

## What was actually executed (not an offline policy import)

Added **opt-in** `verify_immutable_snapshot` as a registered pure SWAAG system tool, requiring a host-only `immutable_snapshot_attestor` runtime capability; it is deliberately absent from default `tools.enabled`. The caller can supply only two attachment IDs, not source state, author identity or trust flags. The external trusted host supplies matching source-version/app/activity/orientation/native dimensions, an independent attestation receipt, and explicit immutable-byte intent. The tool rehashes both actual stored files, inspects PNG IHDR dimensions, uses `decide_bound_immutable_snapshot`, returns PASS/FAIL **only for that exact contract** or REVIEW on unverified identity and never approves the GUI as a whole.

Actual byte-for-byte original PNG screenshots from the author-pinned external RotBench release (`{protocol['input_source_sha256']}`) were ingested through `AgentRuntime.add_attachment`, with two durable `attachment_added` source events in each SWAAG session. One is an author-approved clean screenshot; one is the author-modified version that earlier pixel heuristics missed; a third compares two genuinely valid screenshots of *different activities from the same app*. All original PNG bytes were verified by SHA against frozen prior manifests AND decoded using the actual FFmpeg image decoder. All sessions used a separate SQLite-backed history root under `/data/var/perceptual-independent-20261024/`; not the owner's active SWAAG sessions.

| Real original Android source->final output | Real SWAAG tool verdict | After separate-process saved-session reentry | Tool invocation wall clock |
|---|---|---|---:|
| Exact original PNG source and unchanged final PNG | **{outcomes['clean']['first_real_runtime_tool_verdict'].upper()}** (only immutable byte property) | {outcomes['clean']['restarted_runtime_verdict'].upper()} | {outcomes['clean']['original_tool_dispatch_wall_s']:.4f}s |
| Same source activity with author-injected changed final PNG | **{outcomes['changed']['first_real_runtime_tool_verdict'].upper()}** (only immutable byte property) | {outcomes['changed']['restarted_runtime_verdict'].upper()} | {outcomes['changed']['original_tool_dispatch_wall_s']:.4f}s |
| Real correct screenshot but WRONG Activity/source state | **{outcomes['wrong_activity']['first_real_runtime_tool_verdict'].upper()}** (no automatic reject) | {outcomes['wrong_activity']['restarted_runtime_verdict'].upper()} | {outcomes['wrong_activity']['original_tool_dispatch_wall_s']:.4f}s |

These were actual runtime tool calls, not metadata-only model scores: SWAAG's own ToolRegistry, AttachmentStore content-addressed blobs, session source-event references, `execute_tool_once`, SQLite history and hash-chain events were exercised. Each case recorded one real initial result. A wholly **separate fresh process** then reconstructed each saved session from disk, verified the original SHA and tool result event, attempted the tool with **no** trusted capability (explicit error, *no false success*), independently rebound the trusted host manifest, reexecuted the tool with the same expected result, and archived/rebuilt the complete session. All **3/3** archived/recovered cases retained both source attachments and exactly **two durable tool result events per case**, with **15 journal events per archive**. The final record and all source SHA receipts were written before publishing the report.

## Environment requirement encountered and repaired only in isolated research

Nitro's default `/data/venv` lacked complete OpenTelemetry modules. The available SWAAG Git checkout environment had SQLite **3.50.4**, below SWAAG's real durable-history minimum **3.51.3**. Instead of weakening this invariant or modifying any shared environment, an isolated Python 3.12 test venv was installed under `/data/var/perceptual-independent-20261024/sqlite-test-runtime-20261027/` and official SQLite **3.54.1** was compiled in that same isolated directory (source SHA-256 `{sha(source)}`), loaded **only by the test process**. The actual SWAAG durable-session tests then passed without bypassing the version guard. The SQLite library is not committed or installed globally; no runtime service was restarted.

## Controls, boundaries and remaining work

Unit tests also tested stale source identity, wrong app/activity/viewport, wrong expected file SHA, corrupt persisted bytes, false/absent attestor, model-injected source flags and non-immutable contracts. Those must never produce a definitive GUI failure or approval. A source content hash does not independently prove a screenshot was captured by a particular Android Activity: in these tests an externally frozen publisher manifest was used as research host authority, NOT a trusted live Android device state service. This is a real SWAAG runtime/session integration, **not a production rollout**, and exact-byte diff is not a calibrated detector of clipped text, charts, 3D machine parts or video faults. Source-authentic changing GUIs require human/reference review. Genuine dynamically changing app screenshots, independent human GUI gold, machine geometry/render/photorealism, time-coded natural video + real audio/video sync, and deploying a trusted capture attestor in active SWAAG remain open.

### Preserved receipts and reproduction

Frozen research manifest SHA-256 `{sha(frozen)}`, initial SWAAG real-run result SHA `{sha(executed)}`, recovered/rearchived journal proof SHA `{sha(recovered)}`. Source PNG bytes remain in original isolated research data; test sessions and SQLite journals remain in the dedicated `/data/var` test root and are never published. Full source/replay tests in `tests/research/swaag_real_android_snapshot_session_roundtrip_20261027.py` (`freeze`, `execute`, `recover`) and runtime regression `tests/test_verified_snapshot_tool.py`. The three stages refuse to overwrite recorded results. Because this is a persisted effect test, the already verified source data and archived sessions must **not be run again under the same root**. No scheduled tasks, Google access, trading, production GPU or previously denied message transfers were touched.
''')
    print('REAL_RUNTIME_SOURCE_TO_TARGET_PUBLIC_REPORT',DEST,'sha',sha(DEST))

if __name__=='__main__':main()
