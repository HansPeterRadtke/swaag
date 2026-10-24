from pathlib import Path
import json,hashlib,unittest
from build_context_grounding_v1 import PROJECT,build
from run_context_grounding_v1 import tasks,selector_messages,answer_messages

class AuthSourceSelectionTests(unittest.TestCase):
 def setUp(self):
  self.path=Path(__file__).parent/'context_grounding_v1.json'
  self.dataset=json.loads(self.path.read_text())
 def test_exact_origin_and_answer_frozen(self):
  self.assertEqual(build(),self.dataset)
  self.assertEqual(len(self.dataset['cases']),8)
  for c in self.dataset['cases']:
   self.assertEqual(len(c['passages']),5)
   self.assertEqual(len(set(p['pid'] for p in c['passages'])),5)
   self.assertEqual(set(c['support_ids']),{'r1','r2'})
   self.assertIn(c['gold_answer'],c['answer_options'])
   for p in c['passages']:
    s=(PROJECT/p['file']).read_text('utf8')
    self.assertEqual(s[p['start']:p['end']],p['text'])
    self.assertEqual(hashlib.sha256(s.encode()).hexdigest(),p['file_sha256'])
 def test_context_conditions_preserve_original_question_and_source(self):
  self.assertEqual(len(tasks(self.dataset)),48)
  for c in self.dataset['cases']:
   full=answer_messages(c,{x['pid'] for x in c['passages']})[-1]['content']
   self.assertIn(c['question'],full)
   self.assertEqual(sum(p['text'] in full for p in c['passages']),5)
   selected=selector_messages(c)[-1]['content']
   self.assertEqual(sum(p['text'] in selected for p in c['passages']),5)
if __name__=='__main__':unittest.main()
