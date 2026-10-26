"""Source-grader and frozen oracle-free trigger integrity for real agent tasks.

No benchmark task, model service, GPU or extra architecture is executed.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from copy import deepcopy
import hashlib
import json
import pathlib
import unittest
from evaluate_oracle_free_agent_escalation_20261026 import (
    HERE,ARCHIVE,FIGURE,SOURCE_ZIP_SHA,SOURCE_CSV_SHA,FIGURE_SHA,
    source,transfer_probs,selectors,score_one,ALTS,POLICIES,BUDGET,
)
from analyze_oracle_free_escalation_cluster_effects_20261026 import (
    BASE,OUTCOME_SHA,CASE_SHA,SELECTOR_SHA,source as frozen_uncertainty_source,
)


class OriginalExternallyGradedEscalationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources,cls.version=source()
        cls.report=json.loads((BASE/'policy_final_outcomes_v1.json').read_text())
        cls.observable=json.loads((BASE/'observable_selector_receipts_v1.json').read_text())
        cls.delts=json.loads((BASE/'paired_task_cluster_uncertainty_v1.json').read_text())

    def test_orig_source_hash_and_independent_outcome_version(self):
        self.assertEqual(hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),SOURCE_ZIP_SHA)
        self.assertEqual(hashlib.sha256(FIGURE.read_bytes()).hexdigest(),FIGURE_SHA)
        self.assertEqual(self.version['original_internal_case_csv_sha256'],SOURCE_CSV_SHA)
        self.assertEqual(self.version['original_rows'],1800)
        self.assertEqual(self.version['matched_publisher_configuration_means'],70)
        self.assertEqual(self.version['excluded_nonidentical_model_version_configurations'],20)
        self.assertEqual({x['domain'] for block in self.sources.values() for x in block},{'swebench-verified','terminalbench'})
        self.assertEqual({d:len(v) for d,v in self.sources.items()},{'swebench-verified':140,'terminalbench':140})
        for domain,units in self.sources.items():
            self.assertEqual(len({r['task_id'] for r in units}),20)
            self.assertTrue(all(set(r['alts'])==set(ALTS) for r in units))
            self.assertEqual(len({(r['model'],r['provider']) for r in units}),7)
            self.assertTrue(all(not r['single_agent']['resolved'] or r['single_agent']['has_output'] for r in units))

    def test_frozen_trigger_budget_and_observations_exact(self):
        observed={(o['domain'],o['provider'],o['model'],o['task_id']):o for o in self.observable}
        self.assertEqual(len(observed),280)
        for domain,units in self.sources.items():
            training=self.sources['terminalbench' if domain=='swebench-verified' else 'swebench-verified']
            scores,_=transfer_probs(training,units)
            arms=selectors(units,scores,{})
            self.assertEqual(set(arms),set(POLICIES))
            for p in ('hash_budget_20pct','no_output_then_hash_budget_20pct','cross_benchmark_failure_risk_budget_20pct'):
                self.assertEqual(len(arms[p]),BUDGET)
            self.assertEqual(len(arms['never']),0)
            self.assertEqual(len(arms['all']),140)
            for i,r in enumerate(units):
                item=observed[(domain,r['provider'],r['model'],r['task_id'])]
                self.assertEqual(item['observable_has_output'],r['single_agent']['has_output'])
                self.assertEqual(item['observable_num_steps'],r['single_agent']['num_steps'])
                self.assertAlmostEqual(item['cross_domain_risk_without_target_outcome'],scores[i],places=7)
                for k,selected in item['selected_by'].items():
                    self.assertEqual(selected,i in arms[k])

    def test_unavailable_target_grader_and_alternative_results_cannot_change_deployable_decisions(self):
        # This specifically catches target-label leakage or cherry-picked
        # architecture success when computing a real-time escalation trigger.
        for domain,rows in self.sources.items():
            train=self.sources['terminalbench' if domain=='swebench-verified' else 'swebench-verified']
            original_risk,_=transfer_probs(train,rows)
            original=selectors(rows,original_risk,{})
            mutated=deepcopy(rows)
            for r in mutated:
                r['single_agent']['resolved']=1-r['single_agent']['resolved']
                for arch in ALTS:r['alts'][arch]['resolved']=1-r['alts'][arch]['resolved']
            changed_risk,_=transfer_probs(train,mutated)
            after=selectors(mutated,changed_risk,{})
            self.assertEqual(original_risk,changed_risk)
            for policy in POLICIES:
                if policy!='verified_failure_oracle':self.assertEqual(original[policy],after[policy])
            self.assertNotEqual(original['verified_failure_oracle'],after['verified_failure_oracle'])

    def test_every_offline_final_outcome_equals_blind_replacement_with_original_independent_labels(self):
        results={(r['domain'],r['architecture'],r['policy']):r for r in self.report['benchmark_arms']}
        self.assertEqual(len(results),2*len(ALTS)*len(POLICIES))
        for domain,units in self.sources.items():
            training=self.sources['terminalbench' if domain=='swebench-verified' else 'swebench-verified']
            risk,_=transfer_probs(training,units)
            selections=selectors(units,risk,{})
            for arch in ALTS:
                for policy in POLICIES:
                    result,_=score_one(units,selections[policy],domain,arch,policy)
                    recorded=results[(domain,arch,policy)]
                    for k in ('primary_correct','primary_failures','fallback_workflows','trigger_tp_primary_fail',
                              'trigger_fp_primary_correct','trigger_fn_primary_fail','trigger_tn_primary_correct',
                              'final_correct_after_blind_selection','wrong_to_right_rescues','right_to_wrong_regressions',
                              'wrong_to_wrong_no_improvement','right_to_right_redundant',
                              'original_alt_reported_steps_for_triggered_workflows','net_change_correct_tasks'):
                        self.assertEqual(recorded[k],result[k],(domain,arch,policy,k))
                    if policy=='all':self.assertEqual(recorded['final_correct_after_blind_selection'],sum(r['alts'][arch]['resolved'] for r in units))
                    if policy=='never':self.assertEqual(recorded['final_correct_after_blind_selection'],sum(r['single_agent']['resolved'] for r in units))
                    self.assertEqual(recorded['right_to_wrong_regressions']+recorded['wrong_to_right_rescues']+recorded['right_to_right_redundant']+recorded['wrong_to_wrong_no_improvement'],recorded['fallback_workflows'])

    def test_detectable_no_output_is_truthful_but_incomplete(self):
        b={(r['domain'],r['architecture'],r['policy']):r for r in self.report['benchmark_arms']}
        for d,n,out in [('swebench-verified',77,14),('terminalbench',102,19)]:
            for arch in ALTS:
                row=b[d,arch,'no_output']
                self.assertEqual(row['primary_failures'],n)
                self.assertEqual(row['fallback_workflows'],out)
                self.assertEqual(row['trigger_tp_primary_fail'],out)
                self.assertEqual(row['trigger_fp_primary_correct'],0)
                self.assertEqual(row['right_to_wrong_regressions'],0)
                self.assertEqual(row['trigger_fn_primary_fail'],n-out)
                self.assertEqual(b[d,arch,'no_output_or_at_most_2_steps']['fallback_workflows'],out)

    def test_unconditional_replacement_can_turn_success_into_failure(self):
        matrix={(r['domain'],r['architecture'],r['policy']):r for r in self.report['benchmark_arms']}
        for d,base in [('swebench-verified',63),('terminalbench',38)]:
            for arch in ALTS:
                allrows=matrix[d,arch,'all']
                self.assertEqual(allrows['primary_correct'],base)
                self.assertEqual(allrows['fallback_workflows'],140)
                self.assertGreater(allrows['right_to_wrong_regressions'],0)
                self.assertEqual(allrows['final_correct_after_blind_selection'],base+allrows['net_change_correct_tasks'])

    def test_source_cost_is_workflows_not_fake_GPU_time_or_token_counts(self):
        self.assertEqual(self.report['real_local_Qwen_calls'],0)
        self.assertEqual(self.report['actual_additional_workflows_reexecuted'],0)
        self.assertTrue(self.report['counterfactual_alternative_source_workflows_counted_not_reexecuted'])
        self.assertTrue(any('NOT wall time' in l for l in self.report['limitations']))
        self.assertTrue(all('steps' in str(r) for r in self.report['benchmark_arms']))

    def test_task_grouped_paired_uncertainty_source_pinned(self):
        self.assertEqual(hashlib.sha256((BASE/'policy_final_outcomes_v1.json').read_bytes()).hexdigest(),OUTCOME_SHA)
        self.assertEqual(hashlib.sha256((BASE/'external_task_final_transition_ledger_v1.json').read_bytes()).hexdigest(),CASE_SHA)
        self.assertEqual(hashlib.sha256((BASE/'observable_selector_receipts_v1.json').read_bytes()).hexdigest(),SELECTOR_SHA)
        self.assertEqual(self.delts['source_result_sha256'],OUTCOME_SHA)
        self.assertTrue(self.delts['original_standalone_task_gold_used_only_in_scoring'])
        for row in self.delts['bootstrap_comparisons']:
            self.assertEqual(row['number_original_task_clusters'],20)
            self.assertEqual(row['models_repeated_per_task'],7)
            self.assertEqual(row['number_source_task_model_evaluations'],140)
            self.assertEqual(row['bootstrap_resamples'],1000)
            a=row['paired_delays']['delta_correct']
            self.assertLessEqual(a['descriptive_task_cluster_interval_95pct'][0],a['descriptive_task_cluster_interval_95pct'][1])

if __name__=='__main__':unittest.main()
