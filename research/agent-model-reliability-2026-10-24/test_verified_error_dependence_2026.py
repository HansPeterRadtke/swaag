"""Validate factual invariants of independent published cofailure analysis.

These checks verify primary source provenance and scoring, not LLM reasoning.
"""
import hashlib
import json
import pathlib
import unittest
from analyze_verified_error_dependence_2026 import compute, original_rows

BASE=pathlib.Path(__file__).parent
EVIDENCE=BASE/'evidence'/'matched-compute-architecture'
PRIVATE=pathlib.Path('/data/tmp/agent-model-architecture-source-20261025')

class VerifiedErrorDependenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p=EVIDENCE/'reviewer_role_and_cofailure_exploratory_v1.json'
        cls.result=json.loads(p.read_text())

    def test_source_all_seventy_reproduced(self):
        r=self.result
        self.assertEqual(r['exactly_reconciled_configuration_means'],70)
        self.assertEqual(r['unmatched_configuration_model_identifiers'],20)
        self.assertEqual(r['contradictory_configuration_means'],0)
        self.assertEqual(len(r['verified_task_outcomes']),2)
        for d in r['verified_task_outcomes']:
            self.assertEqual(d['model_task_units'],140)
            self.assertEqual(d['distinct_benchmark_task_ids'],20)
            self.assertEqual(len(d['pairwise_architecture_dependence']),10)
            for c in d['pairwise_architecture_dependence']:
                self.assertEqual(c['both_succeed']+c['first_only']+c['second_only']+c['both_fail'],140)

    def test_undisguised_shared_task_failures(self):
        by={x['dataset']:x for x in self.result['verified_task_outcomes']}
        self.assertEqual(by['swebench-verified']['hard_unsolved_task_count'],4)
        self.assertEqual(by['terminalbench']['hard_unsolved_task_count'],7)
        for d in by.values():
            self.assertEqual(len(d['original_task_ids_unsolved_by_all_7_models_and_all_5_architectures']),d['hard_unsolved_task_count'])
            self.assertEqual(len(d['per_task_independent_model_solutions']),20)
            for id_ in d['original_task_ids_unsolved_by_all_7_models_and_all_5_architectures']:
                self.assertEqual(d['per_task_independent_model_solutions'][id_]['models_all_architectures_failed'],7)

    def test_original_role_assignment_not_reversed(self):
        r={x['family']:x for x in self.result['heterogeneous_reviewer_and_worker_role_results']}
        self.assertEqual([r[f]['strong_subagent_minus_strong_orchestrator_pp'] for f in ('OpenAI','Anthropic','Gemini')],[2,5,5])
        self.assertEqual([r[f]['mixed_peers_vs_strong_only_pp'] for f in ('OpenAI','Anthropic','Gemini')],[3,10,-1])
        self.assertTrue(all(x['centralized']['Low+High']>x['centralized']['High+Low'] for x in r.values()))

    def test_local_immutable_source_recalculation(self):
        archive=PRIVATE/'agent-scaling-v2.1.3.zip'
        het=PRIVATE/'nature-2026-heterogeneous-architecture-original.csv'
        fig=EVIDENCE/'nature-machine-intelligence-2026-source-figs1-2-original.csv'
        if not archive.exists() or not het.exists():self.skipTest('author-source archive not present')
        rebuilt=compute(*original_rows(archive,fig,het))
        self.assertEqual(rebuilt,self.result)

if __name__=='__main__':unittest.main()
