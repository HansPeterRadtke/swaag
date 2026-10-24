#!/usr/bin/env python3
"""Report only SOURCE-LABEL concordance, never human-adjudicated accuracy."""
import argparse,collections,hashlib,json,pathlib,statistics

def analyze(frozen,events):
    frozen_items={x['item_id']:x for x in frozen['items']}
    requests={};stops=[];results=[]
    for e in events:
        event=e['event'];ident=e['trial_id']
        if event=='request_submitted':
            if ident in requests:raise ValueError('duplicate_request')
            case=frozen_items[e['case_id']]
            if e['source_label']!=case['label'] or e['document_sha256']!=case['passage_sha256']:
                raise ValueError('source_label_changed')
            if case['query'] not in e['request']['messages'][-1]['content'] or case['document'] not in e['request']['messages'][-1]['content']:
                raise ValueError('request_lost_exact_evidence')
            requests[ident]=e
        elif event=='response':
            if ident not in requests or any(x['id']==ident for x in results):raise ValueError('unmatched_duplicate_result')
            rq=requests[ident];case=frozen_items[rq['case_id']];score=e['ordinal_score']
            if e['source_label']!=case['label'] or (score is not None and (type(score) is not int or not 0<=score<=8)):
                raise ValueError('invalid_raw_score_or_label')
            if score is not None and e['predicted_binary']!=(score>=5):raise ValueError('binary_rule_changed')
            results.append({'id':ident,'item_id':case['item_id'],'model':rq['model'],'encoding':rq['encoding'],
                            'source_label':case['label'],'rating':score,
                            'source_label_concordance':(score>=5)==bool(case['label']) if score is not None else None,
                            'wall_seconds':e['full_wall_seconds'],'usage':e.get('usage') or {}})
        elif event=='stopped':stops.append(e)
        else:raise ValueError('unknown_event')
    conditions=collections.defaultdict(list)
    pairs=collections.defaultdict(dict)
    paired_encoding=collections.defaultdict(dict)
    for r in results:
        conditions[(r['model'],r['encoding'])].append(r)
        pairs[(r['model'],r['encoding'],r['item_id'].rsplit('-',1)[0])][r['source_label']]=r
        paired_encoding[(r['model'],r['item_id'])][r['encoding']]=r
    return {'source_label_provenance':'ARES DEMONSTRATION DATASET labels unverified as independently human-annotated; observed source-label concordance only',
            'submitted_requests':len(requests),'completed_responses':len(results),'capacity_or_uncertain_stops':len(stops),
            'by_model_encoding':[{'model':k[0],'encoding':k[1],'n':len(v),
               'binary_concordant_source_labels':sum(r['source_label_concordance'] is True for r in v),
               'mean_rating':round(statistics.mean(r['rating'] for r in v if r['rating'] is not None),3) if any(r['rating'] is not None for r in v) else None,
               'mean_full_wall_seconds':round(statistics.mean(r['wall_seconds'] for r in v),3)} for k,v in sorted(conditions.items())],
            'same_query_positive_negative_ranking_pairs':[{'model':key[0],'encoding':key[1], 'query_group':key[2],
                'positive':v[1]['rating'],'negative':v[0]['rating'],
                'positive_greater_than_negative':v[1]['rating']>v[0]['rating'] if v[1]['rating'] is not None and v[0]['rating'] is not None else None} for key,v in pairs.items() if 0 in v and 1 in v],
            'matched_numeric_ordinal_pairs':[{'model':key[0],'item':key[1], 'numeric_score':v['numeric']['rating'],'ordinal_score':v['ordinal']['rating']} for key,v in paired_encoding.items() if 'numeric' in v and 'ordinal' in v],
            'all_results':results}

def main():
    p=argparse.ArgumentParser();p.add_argument('journal',type=pathlib.Path);args=p.parse_args()
    frozen=json.loads((pathlib.Path(__file__).parent/'ares_nq_reference_v1.json').read_text())
    events=[json.loads(x) for x in args.journal.read_text().splitlines() if x.strip()]
    print(json.dumps(analyze(frozen,events),indent=2))
if __name__=='__main__':main()
