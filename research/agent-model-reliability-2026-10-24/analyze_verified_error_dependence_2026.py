#!/usr/bin/env python3
"""Source-verified, exploratory architecture-error dependence and reviewer-role audit.

Original release: Kim et al., Nature Machine Intelligence 2026, Zenodo v2.1.3.
Not pre-registered prior to seeing the original outcomes; not live SWAAG/Qwen.
No output-string voting possible: released per-instance data contain correctness only.
"""
import argparse
import collections
import csv
import hashlib
import io
import json
import math
import pathlib
import statistics
import zipfile

ZIP_SHA='3b6752af8271db7319b1b23dbbe93733bd13e585c87e6f1abc4a17e15b96dc35'
RUN_SHA='c7bde9eac1f28a29397c10ab10e29f0962595e9dd45f17aedc2d309289b12007'
FIG_SHA='402a9fea7430bf4dbf1645c8bafd7850811ffb5bb331568e8088d85c4dc9ffd9'
HET_SHA='df497545c1f007ca8a6c1a9d41eb26602b1060a83f99b0ed0055919653f77c3f'
ARCHS=('single-agent','multi-agent-independent','multi-agent-centralized','multi-agent-decentralized','multi-agent-hybrid')
DSMAP={'swebench-verified':'swe-bench-verified','terminalbench':'terminal-bench'}


def original_rows(archive,figcsv,hetcsv):
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=ZIP_SHA:raise ValueError('modified original Zenodo archive')
    with zipfile.ZipFile(archive) as z:
        names=[n for n in z.namelist() if n.endswith('/scripts/per_instance_results_swe_tb.csv')]
        if len(names)!=1:raise ValueError('ambiguous per-instance source')
        raw=z.read(names[0]);assert hashlib.sha256(raw).hexdigest()==RUN_SHA
    with figcsv.open('rb') as f:figbytes=f.read()
    with hetcsv.open('rb') as f:hetbytes=f.read()
    assert hashlib.sha256(figbytes).hexdigest()==FIG_SHA
    assert hashlib.sha256(hetbytes).hexdigest()==HET_SHA
    runs=list(csv.DictReader(io.StringIO(raw.decode('utf8'))))
    fig=list(csv.DictReader(io.StringIO(figbytes.decode('utf-8-sig'))))
    het=list(csv.DictReader(io.StringIO(hetbytes.decode('utf-8-sig'))))
    assert len(runs)==1800 and len(fig)==260 and len(het)==21
    assert list(runs[0])==['dataset','agent_type','provider','model','instance_idx','resolved','num_steps','success','has_output']
    assert list(het[0])==['architecture','family','configuration','accuracy']
    # The named 'success' column is never set in this particular export.
    # 'resolved' is the evaluated correctness column: independently cross-check
    # all seventy matching model+dataset+architecture means against Figure 1/2.
    assert all(r['success']=='0' and r['resolved'] in ('0','1') for r in runs)
    assert all(r['resolved']!='1' or r['has_output']=='True' for r in runs)
    return runs,fig,het


def phi(a,b,c,d):
    # a=both1,b=onlyA,c=onlyB,d=both0
    n=((a+b)*(a+c)*(d+b)*(d+c))**0.5
    return (a*d-b*c)/n if n else None


def compute(rows,figure,hetero):
    pub=collections.defaultdict(set)
    figmetrics={}
    for x in figure:
        key=(x['dataset'],x['family'],x['model'],x['architecture'])
        figmetrics[key]=float(x['performance'])
        if x['dataset'] in ('swe-bench-verified','terminal-bench'):
            pub[x['dataset']].add((x['family'],x['model']))
    by=collections.defaultdict(list)
    index={}
    for r in rows:
        key=(r['dataset'],r['provider'],r['model'],r['agent_type'])
        by[key].append(r)
        ident=(r['dataset'],r['provider'],r['model'],r['instance_idx'],r['agent_type'])
        if ident in index:raise ValueError('duplicate model/task/architecture record')
        index[ident]=int(r['resolved'])
    if len(index)!=1800 or len(by)!=90:raise ValueError('original source has an unexpected number of tasks')
    matching=unmatched=conflict=0
    for (ds,provider,model,arch),v in by.items():
        if len(v)!=20 or len(set(r['instance_idx'] for r in v))!=20:raise ValueError('source task count mismatch')
        figkey=(DSMAP[ds],provider,model,arch)
        if figkey not in figmetrics:unmatched+=1
        elif abs(statistics.mean(int(r['resolved']) for r in v)-figmetrics[figkey])>1e-9:conflict+=1
        else:matching+=1
    if (matching,unmatched,conflict)!=(70,20,0):raise ValueError('published aggregate outcomes mismatch')
    reports=[]
    for ds in DSMAP:
        models=sorted(set((r['provider'],r['model']) for r in rows if r['dataset']==ds)&pub[DSMAP[ds]])
        assert len(models)==7
        records=[]
        for prov,model in models:
            for i in range(20):
                label=f'{i:04d}'
                records.append({'family':prov,'model':model,'instance_idx':label,
                       'grades':{a:index[(ds,prov,model,label,a)] for a in ARCHS}})
        assert len(records)==140
        correlations=[]
        for ii,first in enumerate(ARCHS):
            for second in ARCHS[ii+1:]:
                ct=collections.Counter((r['grades'][first],r['grades'][second]) for r in records)
                a,b,c,d=(ct[(1,1)],ct[(1,0)],ct[(0,1)],ct[(0,0)])
                correlations.append({'arch_one':first,'arch_two':second,
                 'both_succeed':a,'first_only':b,'second_only':c,'both_fail':d,
                 'paired_discordant_count':b+c,'binary_success_phi':round(phi(a,b,c,d),5) if phi(a,b,c,d) is not None else None})
        pertask={}
        for i in range(20):
            subset=[r for r in records if r['instance_idx']==f'{i:04d}']
            failed_all=sum(not any(x['grades'][a] for a in ARCHS) for x in subset)
            pertask[f'{i:04d}']={'models_evaluated':len(subset),'models_all_architectures_failed':failed_all,
                'single_agent_model_solved':sum(x['grades']['single-agent'] for x in subset)}
        no_solution=[k for k,v in pertask.items() if v['models_all_architectures_failed']==7]
        reports.append({'dataset':ds,'matched_model_versions':[m for p,m in models],
          'model_task_units':len(records),'distinct_benchmark_task_ids':20,
          'pairwise_architecture_dependence':correlations,
          'original_task_ids_unsolved_by_all_7_models_and_all_5_architectures':no_solution,
          'hard_unsolved_task_count':len(no_solution),
          'per_task_independent_model_solutions':pertask})
    hetero_by=collections.defaultdict(dict)
    for r in hetero:
        key=(r['architecture'],r['family'])
        if r['configuration'] in hetero_by[key]:raise ValueError('duplicate role mixture result')
        hetero_by[key][r['configuration']]=float(r['accuracy'])
    mixed=[]
    for fam in ('OpenAI','Anthropic','Gemini'):
        central=hetero_by['centralized',fam];peer=hetero_by['decentralized',fam]
        if set(central)!={'High-only','Low-only','High+Low','Low+High'}:raise ValueError('role variant missing')
        if set(peer)!={'High-only','Low-only','Mixed'}:raise ValueError('peer variant missing')
        mixed.append({'family':fam,'centralized':central,'decentralized':peer,
          'strong_subagent_minus_strong_orchestrator_pp':round(100*(central['Low+High']-central['High+Low']),3),
          'strong_lead_low_worker_vs_strong_only_pp':round(100*(central['High+Low']-central['High-only']),3),
          'low_lead_strong_worker_vs_strong_only_pp':round(100*(central['Low+High']-central['High-only']),3),
          'mixed_peers_vs_strong_only_pp':round(100*(peer['Mixed']-peer['High-only']),3)})
    return {'evidence_type':'EXPLORATORY reanalysis of publisher primary raw per-instance task successes and heterogeneous-team configuration means',
       'original_archive_sha256':ZIP_SHA,'per_instance_source_sha256':RUN_SHA,
       'original_figure_1_2_sha256':FIG_SHA,'original_heterogeneous_figure_2_sha256':HET_SHA,
       'exactly_reconciled_configuration_means':matching,'unmatched_configuration_model_identifiers':unmatched,'contradictory_configuration_means':conflict,
       'verified_task_outcomes':reports,'heterogeneous_reviewer_and_worker_role_results':mixed,
       'limitations':['Binary resolved outcome is an external task-grader result; no candidate text, reviewer critique, or majority-vote output was released here.',
          'Pairwise outcome correlations may reflect common task difficulty, source selection or model capability; they are not calibrated critic-judgment dependence.',
          'Every base task repeats across seven model versions and five architectures; total rows are not independent benchmark instances.',
          'Heterogeneous-team table reports configuration means on BrowseComp-Plus; three families, no task-paired outcomes, no separate latency/token measurements.',
          'SWE-bench and Terminal-Bench source samples include 20 task IDs each; no generalization or automatic routing rule is proven.',
          'The publisher article reports architecture-level communication turn and token overhead but the raw per-instance release does not include full cost.']}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--zenodo-archive',required=True,type=pathlib.Path)
    p.add_argument('--figure-csv',required=True,type=pathlib.Path)
    p.add_argument('--heterogeneity-csv',required=True,type=pathlib.Path)
    p.add_argument('--output',required=True,type=pathlib.Path)
    a=p.parse_args()
    if a.output.exists():raise SystemExit('Refuse to overwrite previously scored evidence')
    report=compute(*original_rows(a.zenodo_archive,a.figure_csv,a.heterogeneity_csv))
    a.output.parent.mkdir(exist_ok=True,parents=True)
    a.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    for ds in report['verified_task_outcomes']:
        print('BENCHMARK',ds['dataset'],'task_model_pairs',ds['model_task_units'],
              'all_architecture_and_model_failure_task_ids',ds['original_task_ids_unsolved_by_all_7_models_and_all_5_architectures'])
        for c in ds['pairwise_architecture_dependence']:
            print('PAIR',c['arch_one'],c['arch_two'],'phi',c['binary_success_phi'],'both_fail',c['both_fail'],'discordant',c['paired_discordant_count'])
    for f in report['heterogeneous_reviewer_and_worker_role_results']:
        print('ROLES',f['family'],'strong_specialist_minus_strong_orchestrator_pp',f['strong_subagent_minus_strong_orchestrator_pp'],
              'decentralized_mixed_minus_strong_only_pp',f['mixed_peers_vs_strong_only_pp'])

if __name__=='__main__':main()
