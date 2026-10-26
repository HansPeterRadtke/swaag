"""Regression + real original Android image evidence for source identity guard."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

REPO=Path(__file__).resolve().parents[2]
REPORT=REPO/'docs/research/perceptual-source-identity-fail-closed-repair-2026-10-26.json'
POLICY=REPO/'src/swaag/perception/acceptance_policy.py'
spec=importlib.util.spec_from_file_location('source_identity_policy_standalone_under_test',POLICY)
P=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=P
spec.loader.exec_module(P)


class SourceIdentityRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r=json.loads(REPORT.read_text())

    def test_independent_prior_source_metadata_and_704_wrong_activity_reviews(self):
        x=self.r
        self.assertEqual(x['source_release_sha256'],'0586b29f82590636806e7dd05a45e6933d46286318a52e444cee4241fb785ca7')
        self.assertEqual(len(x['previous_sources_SHA256']),9)
        self.assertTrue(all(len(s)==64 for s in x['previous_sources_SHA256'].values()))
        ev=x['evidence']
        self.assertEqual(ev['previous_wrong_activity_pairs_flagged_naively'],352)
        self.assertEqual(ev['wrong_reference_unique_author_clean_original_cases'],137)
        self.assertEqual(ev['wrong_reference_actual_author_apps'],28)
        self.assertEqual(ev['wrong_reference_orientation_comparisons'],704)
        self.assertEqual(ev['wrong_reference_flag_after_binding_gate_as_verified_FAIL'],0)
        self.assertEqual(ev['wrong_reference_REVIEW_after_binding_gate'],704)
        self.assertEqual(ev['correct_self_reference_source_byte_contract_PASS'],704)
        self.assertTrue(ev['all_408_distinct_source_artifact_SHA_verified'])
        self.assertEqual(ev['distinct_source_artifacts_reverified'],408)
        bycase=x['per_case_hashed_samples']
        self.assertEqual(len(bycase),704)
        self.assertEqual(len({b['case_digest'] for b in bycase}),137)
        self.assertEqual(set(b['orientation'] for b in bycase),{'portrait','landscape'})
        self.assertTrue(all(b['strict_decision']=='review' and b['same_snapshot_control']=='pass' for b in bycase))

    def test_source_author_image_change_recall_not_misnamed_visual_recognition(self):
        d=self.r['evidence'];xs=self.r['source_modified_100_case_outcomes']
        self.assertEqual(len(xs),100)
        self.assertEqual(len({i['case_sha256'] for i in xs}),100)
        self.assertEqual(d['pinned_author_injected_mutation_snapshot_FAIL'],50)
        self.assertEqual(d['pinned_author_right_unmodified_snapshot_PASS'],50)
        self.assertEqual(d['previous_localized_visual_detector_author_changes_detected'],39)
        self.assertEqual(d['previous_localized_visual_detector_author_changes_missed'],11)
        self.assertEqual(sum(x['source_author_mutant'] and x['exact_byte_contract_verdict']=='fail' for x in xs),50)
        self.assertEqual(sum(not x['source_author_mutant'] and x['exact_byte_contract_verdict']=='pass' for x in xs),50)
        self.assertFalse(self.r['not_claimed']['50_exact_byte_snapshot_change_detections_are_50_human_GUI_clipping_detections'])
        self.assertFalse(self.r['not_claimed']['real_SWAAG_user_facing_workflow_enabled_or_integration_verified'])

    def test_missing_reference_denial_and_bad_metadata_controls_remain_review(self):
        d=self.r
        self.assertEqual(d['evidence']['previous_unavailable_source_reference_cases_REVIEW'],2)
        self.assertTrue(any('HTTP404' in x for x in d['prior_denials_and_incidents']))
        self.assertTrue(any('denied' in x for x in d['prior_denials_and_incidents']))
        self.assertEqual(d['negative_controls']['true_self_snapshot'],'pass')
        self.assertTrue(all(value=='review' for key,value in d['negative_controls'].items() if key!='true_self_snapshot'))
        self.assertFalse(d['performance']['GPU_used'])
        self.assertGreater(d['performance']['CPU_only_wall_seconds'],0)
        self.assertTrue(d['performance']['Nitro_live_resource_snapshot']['other_work_not_stopped'])

    def test_guard_exact_file_sha_and_source_version_vs_wrong_activity(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            approved=root/'approved.png';actual=root/'final.png'
            approved.write_bytes(b'approved-static-native-screen-state-v1')
            actual.write_bytes(approved.read_bytes())
            ahash=hashlib.sha256(approved.read_bytes()).hexdigest()
            orig=P.RenderSourceBinding('android_screen','pinned-source-commit','app.x','MainActivity','portrait',720,1600)
            kwargs=dict(approved_identity=orig,actual_identity=orig,approved_file=approved,actual_file=actual,
                        expected_approved_sha256=ahash,expected_actual_sha256=ahash,
                        reference_manifest_attested=True,rendered_state_attested=True,immutable_snapshot_contract=True)
            self.assertEqual(P.decide_bound_immutable_snapshot(**kwargs),P.Verdict.PASS)
            for bad in (
                P.RenderSourceBinding('android_screen','pinned-source-commit','app.x','SettingsActivity','portrait',720,1600),
                P.RenderSourceBinding('android_screen','other-commit','app.x','MainActivity','portrait',720,1600),
                P.RenderSourceBinding('android_screen','pinned-source-commit','other-app','MainActivity','portrait',720,1600),
                P.RenderSourceBinding('android_screen','pinned-source-commit','app.x','MainActivity','landscape',720,1600),
                P.RenderSourceBinding('android_screen','pinned-source-commit','app.x','MainActivity','portrait',1600,720),
                P.RenderSourceBinding('android_screen','pinned-source-commit','app.x','MainActivity','portrait',True,1600),
            ):
                self.assertEqual(P.decide_bound_immutable_snapshot(**{**kwargs,'actual_identity':bad}),P.Verdict.REVIEW)
            actual.write_bytes(b'approved-static-native-screen-state-v2-CHANGED')
            new_hash=hashlib.sha256(actual.read_bytes()).hexdigest()
            good_updated={**kwargs,'expected_actual_sha256':new_hash}
            self.assertEqual(P.decide_bound_immutable_snapshot(**good_updated),P.Verdict.FAIL)
            self.assertEqual(P.decide_bound_immutable_snapshot(**kwargs),P.Verdict.REVIEW)  # stale actual receipt
            for prop in ('immutable_snapshot_contract','rendered_state_attested','reference_manifest_attested'):
                self.assertEqual(P.decide_bound_immutable_snapshot(**{**good_updated,prop:False}),P.Verdict.REVIEW)
            self.assertEqual(P.decide_bound_immutable_snapshot(**{**good_updated,'expected_approved_sha256':'f'*64}),P.Verdict.REVIEW)
            self.assertEqual(P.decide_bound_immutable_snapshot(**{**good_updated,'approved_file':root/'source_unavailable.png'}),P.Verdict.REVIEW)
            self.assertEqual(P.decide_bound_immutable_snapshot(**{**good_updated,'expected_actual_sha256':'malformed'}),P.Verdict.REVIEW)

    def test_legacy_unbound_verdict_not_mistaken_for_new_bound_evidence(self):
        verified=P.Observation('snapshot_sha_bytes','caller_boolean_only','fail','verified_source',
                               reference_verified=True,actual_output_verified=True,measured_property=True)
        self.assertEqual(P.decide_property([verified]),P.Verdict.FAIL)
        # This caller-only primitive is intentionally retained for compatibility;
        # without exact identities/files, the new strict API must not be invoked.
        self.assertIn('caller-asserted booleans',self.r['cause'])
        self.assertIn('trusted',self.r['repair'])


if __name__=='__main__':unittest.main()
