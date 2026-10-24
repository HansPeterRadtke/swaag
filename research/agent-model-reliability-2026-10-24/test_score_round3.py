import unittest
from logic_oracle import load
from score_round3 import score
from pathlib import Path

class Round3ScorerTest(unittest.TestCase):
    def setUp(self):
        self.case=load(Path(__file__).parent/'formal_holdout_v2.json')[0]
    def make(self,model,answer,method='fused',id_suffix=''):
        id=f'dummy:{model}:{method}{id_suffix}'
        intro={'event':'begin','trial':id,'case':self.case,'method':method,'model':model,'port':14833,'seed':17,'style':'base'}
        output=[{'event':'response','trial':id,'stage':'verdict'} for _ in range(2 if method=='split' else 1)]
        term={'event':'completed','status':'scored','trial':id,'gold':self.case['gold'],'answer':answer,
              'correct':answer==self.case['gold'],'total_elapsed_seconds':1.2,'total_tokens':130}
        return [intro,*output,term]
    def test_same_wrong_answer_is_not_truth(self):
        rows=self.make('m1','yes')+self.make('m2','yes')
        output=score(rows,[self.case])
        self.assertEqual(output['joint_wrong_count'],1)
        self.assertEqual(len(output['cross_model_exact_pairs']),1)
        self.assertEqual(output['scored'],2)
    def test_no_pair_when_split_fails(self):
        rows=self.make('m1','no')
        rows += [{'event':'begin','trial':'broken','case':self.case,'method':'split','model':'m1','port':14833,'seed':17,'style':'base'},
                 {'event':'completed','trial':'broken','status':'invalid_extract','correct':False}]
        output=score(rows,[self.case])
        self.assertEqual(len(output['paired_fused_split']),0)
        self.assertEqual(output['status_counts']['invalid_extract'],1)
    def test_changed_oracle_fails_closed(self):
        rows=self.make('m1','no');rows[0]['case']=dict(self.case,gold='yes')
        with self.assertRaisesRegex(ValueError,'invalid_case_provenance'):score(rows,[self.case])
if __name__=='__main__':unittest.main()
