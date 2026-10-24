import datetime as dt
import json
import pathlib
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(pathlib.Path(__file__).parent))
import run_live

class ExperimentSafetyTests(unittest.TestCase):
    def base_supervisor(self):
        now=dt.datetime.now(dt.timezone.utc).isoformat()
        return {'active_sessions':[], 'backends':{str(p):{'state':'idle','stale':False,'monitor_running':True,'observed_at':now} for p in run_live.PORTS}}

    def probe(self,sup,workers=None,slots=None):
        def rpc(op,*a):
            if op=='orchestration.supervision':return {'supervision':sup}
            if op=='task.list':return {'workers':[] if workers is None else workers}
            raise AssertionError(op)
        with patch.object(run_live,'json_rpc',side_effect=rpc),patch.object(run_live,'fetch',return_value=[{'is_processing':False}] if slots is None else slots):
            return run_live.preflight(14833)

    def test_idle_allows_bounded_probe(self):
        self.assertTrue(self.probe(self.base_supervisor())['idle_slots'])

    def test_active_supervisor_refuses(self):
        sup=self.base_supervisor();sup['active_sessions']=[{'id':1}]
        with self.assertRaisesRegex(RuntimeError,'supervisor_sessions_active'):self.probe(sup)

    def test_busy_backend_refuses(self):
        sup=self.base_supervisor();next(iter(sup['backends'].values()))['state']='processing'
        with self.assertRaisesRegex(RuntimeError,'supervisor_backend_not_verified_idle'):self.probe(sup)

    def test_stale_backend_refuses(self):
        sup=self.base_supervisor();next(iter(sup['backends'].values()))['observed_at']='2026-01-01T00:00:00+00:00'
        with self.assertRaisesRegex(RuntimeError,'supervisor_observation_stale'):self.probe(sup)

    def test_pending_worker_refuses(self):
        with self.assertRaisesRegex(RuntimeError,'production_worker_pending'):self.probe(self.base_supervisor(),workers=[{'status':'queued'}])

    def test_busy_slot_refuses(self):
        with self.assertRaisesRegex(RuntimeError,'llama_slot_busy_or_unknown'):self.probe(self.base_supervisor(),slots=[{'is_processing':True}])

    def test_stage_one_output_survives_later_capacity_refusal(self):
        case={'id':'unit','kind':'extraction','source':'No restart is confirmed.','question':'Did restart happen?','gold':'unknown'}
        trial={'id':'unit:decomposed','kind':'extraction','case':case,'encoding':'decomposed','port':14833,'seed':17}
        first={'value':'Restart unconfirmed','elapsed_seconds':0.2,'raw_message':{'content':'{"facts":"Restart unconfirmed"}'}}
        with patch.object(run_live,'preflight',side_effect=[{'idle_slots':True},RuntimeError('backend_processing')]), patch.object(run_live,'send',return_value=first) as sent:
            out=run_live.run_trial(trial,{14833:'test-id'})
        self.assertEqual(sent.call_count,1)
        self.assertTrue(out['incomplete'])
        self.assertEqual(out['steps'],[first])
        self.assertIn('backend_processing',out['blocked_after_first_stage'])

    def test_plan_is_predefined(self):
        rows=json.loads((pathlib.Path(__file__).parent/'fixtures.json').read_text())
        trials=list(run_live.tasks(rows));self.assertEqual(len(trials),64)
        self.assertEqual(len(set(t['id'] for t in trials)),64)
        self.assertEqual(sum(t['kind']=='relevance' for t in trials),32)
        self.assertEqual(sum(t['kind'] in ('reasoning','extraction') for t in trials),32)

if __name__=='__main__':unittest.main()
