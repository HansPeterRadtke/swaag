"""Basic source equivalence and independence checks for joined rank reports."""
import pathlib
import unittest
from analyze_scifact_cascade_v1 import evaluate

class CascadeBoundaries(unittest.TestCase):
    def test_missing_original_fails_before_any_scoring(self):
        with self.assertRaises(ValueError):
            evaluate({'source_sha256':'bad','cases':[]},pathlib.Path('/not-available'))

if __name__=='__main__':unittest.main()
