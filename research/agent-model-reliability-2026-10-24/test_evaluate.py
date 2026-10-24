import unittest
from evaluate import score

class OracleTests(unittest.TestCase):
 def test_unanimous_wrong_is_not_truth(self):
  actual=score([{'id':'c','kind':'extraction','gold':'no','judgments':[{'answer':'yes'},{'answer':'yes'}]}])[0]
  self.assertEqual(actual['correct'],0); self.assertTrue(actual['unanimous_wrong'])
 def test_invalid_category_not_a_correct_answer(self):
  a=score([{'id':'c','kind':'relevance','gold':8,'judgments':[{'encoding':'ordinal_enum','answer':'highest'}]}])[0]
  self.assertEqual((a['valid'],a['invalid']),(0,1))
 def test_equal_integer_and_category_oracles(self):
  a=score([{'id':'c','kind':'relevance','gold':8,'judgments':[{'encoding':'ordinal_enum','answer':'critical'},{'encoding':'integer_0_8','answer':8}]}])[0]
  self.assertEqual((a['correct'],a['valid']),(2,2))
 def test_no_model_judgments_is_not_success(self):
  a=score([{'id':'c','kind':'reasoning','gold':'unknown','judgments':[]}])[0]
  self.assertEqual((a['n'],a['correct']),(0,0))
if __name__=='__main__': unittest.main()
