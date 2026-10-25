#!/usr/bin/env python3
"""Independently recompute matched-compute SAS/MAS outcome contrasts from publisher CSV.

No inference or label creation. Source is Kim et al., Nature Machine Intelligence
2026, Source Data Figs 1 & 2. Cost figures require the paper's separate tables.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
import pathlib
import statistics

ORIGINAL_SHA256 = '402a9fea7430bf4dbf1645c8bafd7850811ffb5bb331568e8088d85c4dc9ffd9'
EXPECTED_ARCH = ('single-agent', 'multi-agent-independent', 'multi-agent-centralized', 'multi-agent-decentralized', 'multi-agent-hybrid')
DATASET_NAMES = {
    'browsecomp_plus_sampled_100':'BrowseComp-Plus',
    'finance-agent':'Finance Agent',
    'plancraft-test':'PlanCraft',
    'workbench':'WorkBench',
    'swe-bench-verified':'SWE-bench Verified',
    'terminal-bench':'Terminal-Bench',
}
TOL = 1e-10


def parse_source(path):
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != ORIGINAL_SHA256:
        raise ValueError('publisher source version or checksum changed')
    with path.open('r',encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames != ['dataset','architecture','family','model','performance']:
            raise ValueError('unexpected publisher source columns')
        data=list(reader)
    if len(data)!=260:
        raise ValueError(f'expected original two hundred sixty configurations, got {len(data)}')
    expected=set(EXPECTED_ARCH)
    if set(row['architecture'] for row in data)!=expected or set(row['dataset'] for row in data)!=set(DATASET_NAMES):
        raise ValueError('unrecognized architectures or datasets')
    parsed={}
    for row in data:
        key=(row['dataset'],row['family'],row['model'],row['architecture'])
        try:score=float(row['performance'])
        except ValueError as exc: raise ValueError(f'invalid published performance for {key}') from exc
        if key in parsed or not math.isfinite(score) or score<0 or score>1:
            raise ValueError(f'nonunique/bad metric for {key}')
        parsed[key]=score
    groups=collections.defaultdict(dict)
    for (ds,fam,model,arch),score in parsed.items():
        groups[(ds,fam,model)][arch]=score
    if len(groups)!=52 or any(set(v)!=set(EXPECTED_ARCH) for v in groups.values()):
        raise ValueError('matching single-agent/multi-agent architectures incomplete')
    return groups


def sign(delta):
    if delta > TOL:return 'gain'
    if delta < -TOL:return 'loss'
    return 'tie'


def compute(groups):
    matches=[]
    for (dataset,family,model),states in sorted(groups.items()):
        base=states['single-agent']
        choices={architecture:{'performance':states[architecture],
             'delta_pp':round(100*(states[architecture]-base),6),
             'delta_relative_percent':round(100*(states[architecture]/base-1),6) if base>0 else None,
             'direction':sign(states[architecture]-base)}
             for architecture in EXPECTED_ARCH[1:]}
        best=max(choices,key=lambda a:(states[a],-EXPECTED_ARCH.index(a)))
        matches.append({'dataset':dataset,'family':family,'model':model,'single_agent':base,
            'other_architectures':choices,'best_mas':best,
            'best_mas_performance':states[best],
            'best_mas_delta_pp':round(100*(states[best]-base),6),
            'best_mas_status':sign(states[best]-base),
            'above_paper_capability_threshold_45pct':base>0.45})
    by_dataset=[]
    for ds in DATASET_NAMES:
        rows=[x for x in matches if x['dataset']==ds]
        by_arch=[]
        for arch in EXPECTED_ARCH[1:]:
            c=[x['other_architectures'][arch] for x in rows]
            by_arch.append({'architecture':arch,
               'mean_source_performance':round(statistics.mean(x['performance'] for x in c),6),
               'mean_matched_delta_pp':round(statistics.mean(x['delta_pp'] for x in c),4),
               'min_matched_delta_pp':round(min(x['delta_pp'] for x in c),4),
               'max_matched_delta_pp':round(max(x['delta_pp'] for x in c),4),
               'directions':dict(collections.Counter(x['direction'] for x in c))})
        by_dataset.append({'dataset':ds,'name':DATASET_NAMES[ds],'models':len(rows),
           'single_agent_mean':round(statistics.mean(x['single_agent'] for x in rows),6),
           'best_mas_oracle_mean_delta_pp':round(statistics.mean(x['best_mas_delta_pp'] for x in rows),4),
           'best_mas_oracle_directions':dict(collections.Counter(x['best_mas_status'] for x in rows)),
           'by_architecture':by_arch})
    total_arch=[]
    for arch in EXPECTED_ARCH[1:]:
        rows=[x['other_architectures'][arch] for x in matches]
        total_arch.append({'architecture':arch,'pair_count':len(rows),
          'direction_counts':dict(collections.Counter(x['direction'] for x in rows)),
          'macro_mean_matched_delta_pp':round(statistics.mean(x['delta_pp'] for x in rows),4)})
    threshold_report=[]
    for name,subset in [('baseline_at_most_45pct',[x for x in matches if x['single_agent']<=.45]),
                        ('baseline_over_45pct',[x for x in matches if x['single_agent']>.45])]:
        threshold_report.append({'group':name,'n_matched_model_datasets':len(subset),
           'best_mas_gain_count':sum(x['best_mas_status']=='gain' for x in subset),
           'best_mas_tie_count':sum(x['best_mas_status']=='tie' for x in subset),
           'best_mas_loss_count':sum(x['best_mas_status']=='loss' for x in subset),
           'fraction_some_mas_better':round(sum(x['best_mas_status']=='gain' for x in subset)/len(subset),4) if subset else None})
    terminal=[x for x in matches if x['dataset'] in ('swe-bench-verified','terminal-bench')]
    # Re-analyse the reported capability rule as a descriptive comparison, NOT external validation.
    terminal_predict=sum((x['single_agent']>0.45)==(x['best_mas_status']!='gain') for x in terminal)
    return {'source_sha256':ORIGINAL_SHA256,'source_description':'publisher released Fig 1 and 2 configuration means, Nature Machine Intelligence 2026',
        'num_matched_model_dataset_cells':len(matches),'n_original_configurations':260,
        'n_paired_mas_vs_sas_comparisons':len(matches)*4,'macro_source_sampling':'all 52 model-dataset cells equally weighted; no task-level confidence intervals',
        'overall_architecture_contrasts':total_arch,
        'per_dataset_contrasts':by_dataset,
        'best_mas_oracle_overall_direction':dict(collections.Counter(x['best_mas_status'] for x in matches)),
        'best_mas_oracle_average_delta_pp':round(statistics.mean(x['best_mas_delta_pp'] for x in matches),4),
        'baseline_45pct_exploratory_groups':threshold_report,
        'swe_terminal_45pct_descriptive_sign_match':{'n':len(terminal),'matches':terminal_predict,
            'fraction':round(terminal_predict/len(terminal),4),
            'warning':'already derived from same published paper, not new independent prospective validation'},
        'all_original_matched_cells':matches,
        'scientific_limitations':['Publisher CSV contains means per configuration, no case-level trials or paired raw correctness.',
           'Architecture policy effect cannot be separated from implementation choices and realized coordination-turn costs by this CSV alone.',
           'The best-MAS aggregate is post-hoc oracle selection across variants and not deployable without validation.',
           'Model / task cells reuse original datasets; 260 configurations are not 260 independent tasks.',
           'Separate clinical batch paper compared whole-batch single model with isolated workers, not isolated single-model task calls under equal effective context.']}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('source_csv',type=pathlib.Path)
    p.add_argument('--output',required=True,type=pathlib.Path)
    a=p.parse_args()
    if a.output.exists():raise SystemExit('Results already exist; refuse overwrite')
    results=compute(parse_source(a.source_csv))
    a.output.write_text(json.dumps(results,indent=2,ensure_ascii=False,sort_keys=True)+'\n')
    print('CONFIGURATIONS',results['n_original_configurations'],'MATCHED_CELLS',results['num_matched_model_dataset_cells'])
    for s in results['per_dataset_contrasts']:
        print('DATASET',s['name'],'models',s['models'],'SAS_mean',s['single_agent_mean'],'MAS_best_oracle_directions',s['best_mas_oracle_directions'])
        for m in s['by_architecture']:
            print('  ARCH',m['architecture'],'mean_matched_delta_pp',m['mean_matched_delta_pp'],'directions',m['directions'])
    print('THRESHOLD',results['baseline_45pct_exploratory_groups'])
    print('SWE_TERMINAL',results['swe_terminal_45pct_descriptive_sign_match'])

if __name__=='__main__':main()
