import pathlib,json,unittest
from score_ares_reference import analyze

class SourceLabelConcordanceChecks(unittest.TestCase):
 def setUp(self):
  self.dataset=json.loads((pathlib.Path(__file__).parent/'ares_nq_reference_v1.json').read_text());self.c=self.dataset['items'][0]
 def request(self):
  c=self.c;return {'event':'request_submitted','trial_id':'t','case_id':c['item_id'],'source_label':c['label'],
    'document_sha256':c['passage_sha256'],'model':'m','encoding':'numeric',
    'request':{'messages':[{'content':c['query']+' '+c['document']}]}}
 def test_one_result_records_source_label_not_accuracy(self):
  req=self.request();out={'event':'response','trial_id':'t','ordinal_score':2,'predicted_binary':False,'source_label':self.c['label'],'full_wall_seconds':1.0,'usage':{}}
  a=analyze(self.dataset,[req,out]);self.assertEqual(a['completed_responses'],1)
  self.assertIn('unverified',a['source_label_provenance'])
 def test_changed_external_gold_rejected(self):
  a=self.request();a['source_label']=int(not bool(self.c['label']))
  with self.assertRaisesRegex(ValueError,'source_label_changed'):analyze(self.dataset,[a])
if __name__=='__main__':unittest.main()
