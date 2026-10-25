"""Validate original publisher source and paired comparisons; not a model-accuracy test."""
import pathlib
import unittest

from reanalyze_matched_compute_architecture_2026 import parse_source, compute

ROOT=pathlib.Path(__file__).resolve().parent/'evidence'/'matched-compute-architecture'


class MatchedComputeSourceReanalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source=ROOT/'nature-machine-intelligence-2026-source-figs1-2-original.csv'
        cls.groups=parse_source(cls.source)
        cls.results=compute(cls.groups)

    def test_matched_control_and_full_source_coverage(self):
        self.assertEqual(len(self.groups),52)
        self.assertEqual(self.results['n_original_configurations'],260)
        self.assertEqual(self.results['n_paired_mas_vs_sas_comparisons'],208)
        self.assertTrue(all(len(v)==5 and 'single-agent' in v for v in self.groups.values()))

    def test_published_domain_means_are_reproduced_without_golden_output_leak(self):
        by={x['dataset']:x for x in self.results['per_dataset_contrasts']}
        finance=by['finance-agent']
        plancraft=by['plancraft-test']
        workbench=by['workbench']
        finarch={x['architecture']:x for x in finance['by_architecture']}
        planarch={x['architecture']:x for x in plancraft['by_architecture']}
        workarch={x['architecture']:x for x in workbench['by_architecture']}
        self.assertAlmostEqual(finance['single_agent_mean'],0.348889,places=6)
        self.assertAlmostEqual(finarch['multi-agent-centralized']['mean_source_performance'],0.631111,places=6)
        self.assertAlmostEqual(plancraft['single_agent_mean'],0.567823,places=6)
        self.assertAlmostEqual(planarch['multi-agent-independent']['mean_matched_delta_pp'],-39.7542,places=4)
        self.assertAlmostEqual(workarch['multi-agent-decentralized']['mean_matched_delta_pp'],3.5556,places=4)

if __name__=='__main__': unittest.main()
