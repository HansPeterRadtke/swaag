import json
import pathlib
import tempfile
import unittest
from unittest.mock import patch
import run_round3 as r3
from logic_oracle import load

class ThirdRoundTests(unittest.TestCase):
    def test_plan_identity_and_oracles(self):
        c=load(pathlib.Path(__file__).parent/'formal_holdout_v2.json')
        a=list(r3.plan(c))
        self.assertEqual(len(a),48)
        self.assertEqual(len({x['trial'] for x in a}),48)
        self.assertEqual(sum(x['method']=='split' for x in a),12)
        self.assertEqual(sum(x['style']=='paraphrase' for x in a),12)
    def test_preflight_refuses_before_issuing(self):
        c=load(pathlib.Path(__file__).parent/'formal_holdout_v2.json')[0]
        p=dict(next(r3.plan([c])),model='fake')
        with tempfile.TemporaryDirectory() as d,patch.object(r3,'preflight',side_effect=RuntimeError('busy')):
            with patch.object(r3,'generate') as g:
                with self.assertRaisesRegex(RuntimeError,'busy'):r3.run_one(p,pathlib.Path(d)/'j.jsonl')
                g.assert_not_called()
    def test_completed_fused_logs_schema_and_original(self):
        c=load(pathlib.Path(__file__).parent/'formal_holdout_v2.json')[0]
        p=dict(next(r3.plan([c])),model='fake')
        result={'request':{},'raw_response':{'message':{'content':'{"answer":"no"}'}},'usage':{'total_tokens':9},'elapsed_seconds':0.1,'value':'no','valid':True}
        with tempfile.TemporaryDirectory() as d,patch.object(r3,'preflight',return_value={'idle':True}),patch.object(r3,'generate',return_value=result):
            out=pathlib.Path(d)/'j.jsonl'; self.assertEqual(r3.run_one(p,out),'correct')
            rows=[json.loads(z) for z in out.read_text().splitlines()]
            self.assertEqual([a['event'] for a in rows],['begin','request_start','response','completed'])
            self.assertEqual(rows[-1]['total_tokens'],9)
    def test_aborted_second_stage_preserves_first_stage(self):
        c=load(pathlib.Path(__file__).parent/'formal_holdout_v2.json')[0]
        p=dict(list(r3.plan([c]))[1],model='fake')
        fake={'request':{},'raw_response':{},'usage':{'total_tokens':5},'elapsed_seconds':0.1,'value':'trusted facts','valid':True}
        with tempfile.TemporaryDirectory() as d, patch.object(r3,'preflight',side_effect=[{'idle':True},RuntimeError('production busy')]),patch.object(r3,'generate',return_value=fake) as g:
            out=pathlib.Path(d)/'j.jsonl'
            with self.assertRaisesRegex(RuntimeError,'production busy'):r3.run_one(p,out)
            self.assertEqual(g.call_count,1)
            rows=[json.loads(z) for z in out.read_text().splitlines()]
            self.assertEqual(rows[-1]['event'],'response')
            self.assertEqual(rows[-1]['stage'],'facts')
if __name__=='__main__':unittest.main()
