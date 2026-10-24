#!/usr/bin/env python3
"""Extract a frozen balanced public NQ relevance reference without relabeling.

Upstream labels are source-provided; their HUMAN provenance is NOT verified.
The full upstream TSV is kept outside the repository and never copied to Thor.
"""
import argparse,collections,hashlib,json,pathlib,random,re

SOURCE_SHA='649321a57e4dfe404d43c9ecb2443889b89cdff12e5ec62cca04cc42602bf236'
SOURCE_URL='https://raw.githubusercontent.com/stanford-futuredata/ARES/main/datasets/example_files/nq_labeled_output.tsv'
SEED=20261025

def words(s):return set(re.findall(r'[a-z0-9]{3,}',s.lower()))
def relevance_overlap(q,doc):
    a,b=words(q),words(doc)
    return len(a&b)/len(a) if a else 0

def build(path):
    import pandas as pd
    blob=path.read_bytes()
    if hashlib.sha256(blob).hexdigest()!=SOURCE_SHA:raise ValueError('unexpected_upstream_data_version')
    d=pd.read_csv(path,sep='\t',dtype=str,keep_default_na=False,low_memory=False)
    assert len(d)==6189
    counts=collections.Counter(d.Context_Relevance_Label)
    assert counts=={'1.0':2653,'0.0':1768,'':1768}
    eligible=collections.defaultdict(lambda:{'0.0':[],'1.0':[]})
    for i,row in d.iterrows():
        lab=row['Context_Relevance_Label']
        if lab not in ('0.0','1.0'):continue
        q=row['Query'].strip();doc=row['Document'].strip()
        if not 12<=len(q)<=175 or not 120<=len(doc)<=1350:continue
        if '\x00' in doc or len(words(q))<3:continue
        eligible[q][lab].append((i,doc))
    viable=[]
    for q,bins in eligible.items():
        if not all(bins[x] for x in ('0.0','1.0')):continue
        # Select the hardest (most question-topic-overlapping) negative passage
        # for each query, but choose queries evenly across overlap strata.
        neg=max(bins['0.0'],key=lambda row:(relevance_overlap(q,row[1]),-row[0]))
        pos=min(bins['1.0'],key=lambda row:row[0])
        if pos[1]==neg[1]:continue
        viable.append((q,pos,neg,relevance_overlap(q,neg[1])))
    viable.sort(key=lambda x:(x[3],x[0]))
    rng=random.Random(SEED)
    third=len(viable)//3
    chosen=[]
    for group in (viable[:third],viable[third:2*third],viable[2*third:]):
        if len(group)<8:raise ValueError('not_enough_distinct_queries')
        chosen+=rng.sample(group,8)
    rng.shuffle(chosen)
    cases=[]
    for j,(q,pos,neg,overlap) in enumerate(chosen):
        for label,row in [(1,pos),(0,neg)]:
            index,document=row
            cases.append({'item_id':f'nq-{j:02d}-{label}', 'query':q,'document':document,
                          'label':label, 'source_row_index':int(index),
                          'passage_sha256':hashlib.sha256(document.encode('utf8')).hexdigest(),
                          'question_token_overlap_with_negative':round(overlap,6),
                          'source_label_type':'unverified source-provided example dataset label; NOT independently established as human-rater ground truth'})
    return {'source_url':SOURCE_URL,'source_sha256':SOURCE_SHA,'original_rows':len(d),
            'source_label_counts':dict(counts),'label_provenance_caution':'ARES example output appears constructed/mixed; do not claim the full corpus has independent blinded human relevance ratings; this external set is supplemental and not SWAAG-context representative',
            'sampling':{'version':'sample-ares-nq-v1','seed':SEED,'queries':24,'pairs':48,'per_overlap_tercile':8,'filters':'query 12..175 chars, >=3 words; full document 120..1350 chars; only source 0.0 and 1.0 labels; choose maximum-overlap negative per query then stratify query overlap thirds'},
            'items':cases}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=pathlib.Path,required=True);parser.add_argument('--output',type=pathlib.Path,required=True)
    a=parser.parse_args()
    if a.output.exists():raise SystemExit('Frozen output exists: refuse overwrite')
    out=build(a.input)
    a.output.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
    print('pairs',len(out['items'])//2,'items',len(out['items']),'saved_bytes',a.output.stat().st_size,'unique_queries',len({x['query'] for x in out['items']}),'avg_doc_chars',round(sum(len(x['document']) for x in out['items'])/len(out['items'])))
