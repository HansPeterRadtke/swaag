"""Offline companion validation of persisted, separately reentered real SWAAG receipts."""
from pathlib import Path
import json
import unittest

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'docs/research/perceptual-real-swaag-attachment-snapshot-roundtrip-2026-10-27.json'


class RealAndroidSwaagSavedStateProof(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=json.loads(REPORT.read_text())
        cls.cases=cls.data['real_source_to_target_end_state_results']

    def test_source_hash_status_and_registered_tool_scope(self):
        d=self.data
        self.assertEqual(d['original_screenshot_source_release_SHA256'], '0586b29f82590636806e7dd05a45e6933d46286318a52e444cee4241fb785ca7')
        self.assertTrue(d['tool_registered_in_real_SWAAG_ToolRegistry'])
        self.assertTrue(d['tool_default_disabled_and_host_capability_required'])
        self.assertEqual(d['new_registered_tool'],'verify_immutable_snapshot')
        self.assertFalse(d['model_inference_or_gpu_call'])
        self.assertEqual(set(self.cases),{'clean','changed','wrong_activity'})
        for name in ('pinned_source_manifest_SHA256','first_runtime_execution_SHA256','separate_restarted_recovery_SHA256'):
            self.assertEqual(len(d[name]),64)

    def test_verified_exact_property_and_bad_state_review(self):
        self.assertEqual({k:v['first_real_runtime_tool_verdict'] for k,v in self.cases.items()},
                         {'clean':'pass','changed':'fail','wrong_activity':'review'})
        self.assertEqual({k:v['restarted_runtime_verdict'] for k,v in self.cases.items()},
                         {'clean':'pass','changed':'fail','wrong_activity':'review'})
        self.assertEqual(self.cases['clean']['original_author_output_source_state_sha256'],self.cases['clean']['actual_user_visible_PNG_source_sha256'])
        self.assertNotEqual(self.cases['changed']['original_author_output_source_state_sha256'],self.cases['changed']['actual_user_visible_PNG_source_sha256'])
        self.assertNotEqual(self.cases['wrong_activity']['source_original_state_from_pinned_author'],self.cases['wrong_activity']['target_original_state_from_pinned_author'])
        self.assertTrue(all(x['not_general_Graphics_GUI_visual_quality_verdict'] for x in self.cases.values()))

    def test_real_source_boundaries_and_actual_decoder(self):
        for c in self.cases.values():
            self.assertTrue(all(isinstance(k,int) and k>0 for k in c['source_image_native_size_wh']))
            self.assertTrue(all(isinstance(k,int) and k>0 for k in c['target_image_native_size_wh']))
            self.assertEqual(c['source_image_native_size_wh'],c['target_image_native_size_wh'])
            self.assertEqual(len(c['raw_png_decode_wall_seconds']),2)
            self.assertTrue(all(v>0 for v in c['raw_png_decode_wall_seconds']))

    def test_actual_runtime_history_saved_and_separate_process_reentry(self):
        ids=[]
        for c in self.cases.values():
            self.assertTrue(c['source_attestor_missing_after_restart_refused'])
            self.assertEqual(c['persisted_result_event_count'],2)
            self.assertEqual(c['saved_journal_archive_event_count'],15)
            self.assertTrue(c['archive_reentry_kept_attachments'])
            self.assertEqual(len(c['original_first_tool_result_event_SHA256']),64)
            self.assertEqual(len(c['rebound_tool_result_event_SHA256']),64)
            self.assertNotEqual(c['original_first_tool_result_event_SHA256'],c['rebound_tool_result_event_SHA256'])
            ids.append(c['session_id_SHA256'])
        self.assertEqual(len(set(ids)),3)
        self.assertIn('SQLite',self.data['durable_storage'])

    def test_unvalidated_caller_security_boundary_not_misrepresented(self):
        limit=self.data['verified_host_manifest_forgery_limit']
        self.assertIn('not a live Android Activity',limit)
        self.assertTrue(any('rather than a live android' in line.lower() for line in self.data['scope_and_unproved']))
        self.assertTrue(any('not enable this opt-in' in line for line in self.data['scope_and_unproved']))
        self.assertTrue(any('audio' in line.lower() for line in self.data['scope_and_unproved']))

    def test_sqlite_requirement_and_current_runtime_resource(self):
        p=self.data['resource']
        self.assertEqual(p['sqlite_runtime_version'],'3.54.1')
        self.assertTrue(p['local_SWAAG_runtime_venv_not_system_or_production'])
        self.assertGreater(p['runtime_tool_s_min'],0)
        self.assertLess(p['runtime_tool_s_max'],5.)
        self.assertTrue(p['active_GPU_models_unchanged'])
        self.assertFalse(self.data['model_inference_or_gpu_call'])

if __name__=='__main__':unittest.main()
