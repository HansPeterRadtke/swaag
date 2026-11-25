"""Deterministic tests of the recorded pilot evidence, not inference."""
from pathlib import Path
import copy
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/demos'))
try:
    import verify_scicom_balanced_pilot_evidence as audit
finally:
    sys.path.pop(0)


def case(lang='en',label='hold',idx=0):
    return dict(id=f'{lang}-{label}-{idx}',language=lang,wav_sha256='fixture',
                index=0,label=label,start_sample=0,end_sample=16000,decision_sample=5120)


def observed(c,score=.5):
    return {**c,'feature_seconds':.02,
            'models':{n:{'score':score,'inference_seconds':.1} for n in audit.PINS}}


class EvidenceTests(unittest.TestCase):
    def test_exact_receipt_and_tampered_receipts(self):
        c=case();r=observed(c)
        audit.validate_decision(c,r)
        edits=(lambda x:x.update(decision_sample=7680),
               lambda x:x.update(label='eot'),
               lambda x:x['models'].pop('small'),
               lambda x:x['models']['tiny'].update(score=float('nan')),
               lambda x:x['models']['tiny'].update(score=1.1),
               lambda x:x.update(feature_seconds=-1),
               lambda x:x['models']['base'].update(inference_seconds=True))
        for edit in edits:
            with self.subTest(edit=edit):
                bad=copy.deepcopy(r);edit(bad)
                with self.assertRaises(ValueError):
                    audit.validate_decision(c,bad)

    def test_exact_balance_and_no_missing_score(self):
        rows=[observed(case(l,k,i),.9 if k=='eot' else .1)
              for l in ('en','de') for k in ('hold','eot') for i in range(40)]
        m=audit.metrics(rows)
        self.assertTrue(all(v[l]['passes_primary_research_screen']
                            for v in m.values() for l in ('en','de')))
        rows[0]['models']['tiny']['score']=.5
        self.assertFalse(audit.metrics(rows)['tiny']['en']['passes_primary_research_screen'])
        with self.assertRaises(ValueError):
            audit.metrics(rows[:-1])

    def test_deterministic_sampling_excludes_old_and_never_duplicates(self):
        rows=[]
        for lang in ('en','de'):
            for i in range(100):
                rows.append(dict(id=f'{lang}-{i}',language=lang,wav_sha256='fixture',
                                 spans=[dict(index=0,label='hold',start_sample=0,end_sample=8000,decision_sample=5120),
                                        dict(index=1,label='eot',start_sample=16000,end_sample=26000,decision_sample=20480)]))
        excluded=['en-1','de-2']
        a=audit.build_cases(rows,excluded)
        self.assertEqual(a,audit.build_cases(list(reversed(rows)),excluded))
        self.assertEqual(len(a),160)
        self.assertEqual(len({x['id'] for x in a}),160)
        self.assertFalse(set(excluded)&{x['id'] for x in a})

    def test_pinned_file_rejects_changes_and_symlinks(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'proof';p.write_bytes(b'original')
            self.assertEqual(audit.pinned(root,'proof',audit.sha(b'original')),p)
            with self.assertRaises(ValueError):
                audit.pinned(root,'proof',audit.sha(b'changed'))
            (root/'link').symlink_to(p)
            with self.assertRaises(ValueError):
                audit.pinned(root,'link',audit.sha(b'original'))


if __name__=='__main__':unittest.main()
