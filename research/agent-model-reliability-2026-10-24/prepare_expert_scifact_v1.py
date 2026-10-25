#!/usr/bin/env python3
"""Reconstruct reproducible *expert-labelled* SciFact claim+abstract cases.

Do not commit copyrighted scientific abstract text or original full dataset to
public Git. Only frozen IDs, evidence sentence indices, labels and hashes are
committed; private source text is regenerated under a local scratch directory.
"""
import argparse,hashlib,io,json,pathlib,random,tarfile
from collections import Counter,defaultdict

UPSTREAM_SHA='11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be'
UPSTREAM_URL='https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz'
SEED=20261025

def package(c):
    return c['title'].strip()+'\n'+'\n'.join(c['abstract'])

def sha(data):return hashlib.sha256(data.encode('utf8')).hexdigest()

def source_examples(archive):
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=UPSTREAM_SHA:raise ValueError('SciFact upstream version or hash changed')
    with tarfile.open(archive,'r:gz') as tf:
        corpus={str(c['doc_id']):c for c in map(json.loads,io.TextIOWrapper(tf.extractfile('data/corpus.jsonl'),encoding='utf8'))}
        rows=list(map(json.loads,io.TextIOWrapper(tf.extractfile('data/claims_dev.jsonl'),encoding='utf8')))
    grouped={}
    for claim in rows:
        for doc_id,evidence in (claim.get('evidence') or {}).items():
            labels={r['label'] for r in evidence}
            if len(labels)!=1 or not labels <= {'SUPPORT','CONTRADICT'}:raise ValueError('inconsistent source evidence labels')
            doc=corpus[str(doc_id)]
            indices=sorted({int(i) for item in evidence for i in item['sentences']})
            if not indices or min(indices)<0 or max(indices)>=len(doc['abstract']):raise ValueError('invalid rationale index')
            text=package(doc)
            grouped[(int(claim['id']),str(doc_id))]={'source_claim_id':int(claim['id']),'doc_id':str(doc_id),
              'claim':claim['claim'],'title':doc['title'],'sentences':doc['abstract'],
              'label':next(iter(labels)),'expert_rationale_sentence_indices':indices,
              'claim_sha256':sha(claim['claim']),'abstract_sha256':sha(text),'abstract_length_chars':len(text)}
    return list(grouped.values())

def choose(cases):
    # Split by source label and literal abstract length before seeing model outputs.
    selected=[];rng=random.Random(SEED)
    for label in ('SUPPORT','CONTRADICT'):
        eligible=[x for x in cases if x['label']==label and 500<=x['abstract_length_chars']<=3600 and len(x['claim'])<=260]
        eligible.sort(key=lambda x:(x['abstract_length_chars'],x['source_claim_id'],x['doc_id']))
        cut=len(eligible)//3
        for chunk in (eligible[:cut],eligible[cut:2*cut],eligible[2*cut:]):
            if len(chunk)<8:raise ValueError('too few source expert annotations')
            selected+=rng.sample(chunk,8)
    rng.shuffle(selected)
    assert len(selected)==48 and len({(x['source_claim_id'],x['doc_id']) for x in selected})==48
    return selected

def build(archive):
    selected=choose(source_examples(archive))
    public={'study':'scifact-expert-claims-v1','upstream_url':UPSTREAM_URL,'upstream_sha256':UPSTREAM_SHA,
      'split':'dev','seed':SEED,'selection':'24 SUPPORT +24 CONTRADICT expert-labelled claim/abstract pairs; stratify within each label by original abstract character-length thirds; 8 samples per third; length filter 500..3600, claim <=260; evidence rationale indices union',
      'provenance':'expert-written scientific claims with annotated support/refute evidence and rationale sentences; SciFact Wadden et al. EMNLP 2020; not general task-relevance ground truth; original source may occur in model pretraining',
      'license_note':'original SciFact distribution may have noncommercial license; full scientific abstracts and claims are held only in private scratch, not included in public Git',
      'cases':[]}
    private=[]
    for i,x in enumerate(selected):
        id_=f'scifact-dev-v1-{i:02d}'
        public['cases'].append({'case_id':id_,'source_claim_id':x['source_claim_id'],'source_doc_id':x['doc_id'],
          'gold':x['label'],'claim_sha256':x['claim_sha256'],'abstract_sha256':x['abstract_sha256'],
          'rationale_indices':x['expert_rationale_sentence_indices'],'abstract_chars':x['abstract_length_chars']})
        private.append({'case_id':id_,'claim':x['claim'],'title':x['title'],'sentences':x['sentences'],
          'rationale_indices':x['expert_rationale_sentence_indices'],'gold':x['label'],
          'source_claim_id':x['source_claim_id'],'source_doc_id':x['doc_id'],
          'claim_sha256':x['claim_sha256'],'abstract_sha256':x['abstract_sha256']})
    return public,{'study':'scifact-expert-claims-v1','original_archive_sha256':UPSTREAM_SHA,'cases':private}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=pathlib.Path,required=True);p.add_argument('--manifest',type=pathlib.Path,required=True);p.add_argument('--private',type=pathlib.Path,required=True)
    args=p.parse_args();out,private=build(args.archive)
    if args.manifest.exists() or args.private.exists():raise SystemExit('Refuse overwrite frozen output')
    args.manifest.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    args.private.parent.mkdir(parents=True,exist_ok=True)
    args.private.write_text(json.dumps(private,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print('cases',len(out['cases']),'labels',dict(Counter(x['gold'] for x in out['cases'])),
         'public_manifest_bytes',args.manifest.stat().st_size,'private_data_bytes',args.private.stat().st_size)
