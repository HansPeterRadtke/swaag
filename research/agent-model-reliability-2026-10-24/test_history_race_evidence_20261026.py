"""Check persisted source-pinned, isolated JSONL partial-append evidence only."""
import hashlib
import json
import subprocess
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

class HistoryRaceEvidence(unittest.TestCase):
    def test_source_identity_and_transient_vs_persistent_distinction(self):
        r=json.loads((HERE/'evidence/final-foreground-receipts/history_partial_append_v1.json').read_text())
        # This is evidence of the PRE-FIX source, not a moving assertion that
        # today's patched file must still match the old vulnerable version.
        # Keep the historical SHA and observation exactly as originally saved.
        original = subprocess.run(
            ['git', 'show', '5a0ffde868bac6290bccbe4d045abcab6cdbedac:src/swaag/history.py'],
            cwd=ROOT, check=True, capture_output=True,
        ).stdout
        self.assertEqual(r['canonical_history_py_sha256'],hashlib.sha256(original).hexdigest())
        self.assertTrue(r['withheld_write_tail_deterministically'])
        self.assertEqual(r['reader_error_during_split_append'],'HistoryCorruptionError')
        self.assertTrue(r['reader_recovered_after_append_complete'])
        self.assertTrue(r['final_history_chain_verifies'])
        self.assertGreater(r['complete_history_size'],r['temporary_partial_history_size'])
        self.assertTrue(r['no_production_history_files_touched'])
        self.assertTrue(r['no_model_calls_or_external_tool_effects'])
        self.assertIn('not measured spontaneous failure frequency',r['interpretation'])

if __name__=='__main__':unittest.main()
