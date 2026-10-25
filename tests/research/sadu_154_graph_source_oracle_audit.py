#!/usr/bin/env python3
"""Independently cross-check SADU human diagram QA labels against graph source.

154 human-authored original graph JSONs, source 154 QA files and 154 REAL images
in pinned 2026 SADU Zenodo ZIP. Recompute six well-defined property counts
without using benchmark authors' code or VLM outputs; do not execute scripts.

This checks numeric QA reference internal CONSISTENCY, not screenshot rendered
correctness (the source diagram image could still be wrong visually). Avoid
pseudo-tests that inflate software counts; every individual mismatch retained.
"""
from pathlib import Path
import zipfile,json,hashlib,collections,datetime
ZIP=Path('/data/var/perceptual-independent-20261024/sadu-diagrams/SADU_ASE_submission.zip')
OUT=ZIP.parent/'sadu_154_independent_source_graph_QA_audit_20261025.json'
RULES={
 'total_entity_number':'number of original entity graph entries',
 'total_relation_number':'number of relation graph entries',
 'total_cluster_number':'number of cluster graph entries',
 'how_many_entities_not_in_clusters':'entity keys absent from ALL include_entity lists (nested or outer; union)',
 'how_many_relations_have_label':'relations whose original name string is not empty/whitespace',
 'how_many_clusters_have_clusters_within':'clusters with nonempty include_cluster list',
}

def compute(g):
 ent=g['entity'];rel=g['relation'];clusters=g['cluster']
 union={k for c in clusters.values() for k in c.get('include_entity',[])}
 return {'total_entity_number':len(ent),'total_relation_number':len(rel),'total_cluster_number':len(clusters),'how_many_entities_not_in_clusters':len(set(ent)-union),'how_many_relations_have_label':sum(bool(str(edge.get('name','')).strip()) for edge in rel),'how_many_clusters_have_clusters_within':sum(bool(c.get('include_cluster',[])) for c in clusters.values())}

def main():
 if OUT.exists():print('PREVIOUS_SOURCE_ORACLE_RECONCILIATION_PRESERVED');return
 results=[]
 with zipfile.ZipFile(ZIP) as z:
  files=set(z.namelist());targets=sorted(n for n in files if n.startswith('SADU-main/dataset/SAD/') and '/json_object/' in n and n.endswith('.json'))
  assert len(targets)==154
  for n in targets:
   parts=n.split('/');domain=parts[3];base=parts[-1][:-5];qa=n.replace('/json_object/','/QA/').replace('.json','_QA.json')
   assert qa in files
   image=f'SADU-main/dataset/SAD/{domain}/Diagram/{base}.png';assert image in files
   graph=json.loads(z.read(n));gold=json.loads(z.read(qa));calc=compute(graph)
   found={r['metadata']['sub_type']:r for r in gold}
   results_here=[]
   for field,expected in calc.items():
    assert field in found
    answer=found[field]['answer']
    check={'subtype':field,'independent_graph_calculated':expected,'source_author_qa_answer':answer,'match':answer==[expected]}
    results_here.append(check)
   assert len(results_here)==len(RULES)
   results.append({'source_domain':domain,'source_diagram_id':base,'source_graph_sha256':hashlib.sha256(z.read(n)).hexdigest(),'source_QA_sha256':hashlib.sha256(z.read(qa)).hexdigest(),'image_source_sha256':hashlib.sha256(z.read(image)).hexdigest(),'checks':results_here})
 summary={k:{'n':len(results),'matches':sum(next(x for x in r['checks'] if x['subtype']==k)['match'] for r in results),'disagreements':sum(not next(x for x in r['checks'] if x['subtype']==k)['match'] for r in results)} for k in RULES}
 mismatches=[{'domain':r['source_domain'],'diagram':r['source_diagram_id'],**check} for r in results for check in r['checks'] if not check['match']]
 report={'source':'SADU 2026 real software architecture diagrams and author manually assembled graph structures, source original QA answers','date_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'archive_sha256':hashlib.sha256(ZIP.read_bytes()).hexdigest(),'diagram_count':len(results),'numeric_QA_invariants_tested':len(results)*len(RULES),'property_definitions':RULES,'results':summary,'individual_mismatches':mismatches,'source_receipts_by_diagram':[{k:r[k] for k in ('source_domain','source_diagram_id','source_graph_sha256','source_QA_sha256','image_source_sha256')} for r in results],'caveat':'The benchmark QA numeric labels are deterministic facts derived from author graph annotations; re-verifying those facts does NOT establish that the original rendered PNG accurately shows the annotated nodes/edges, which requires visual checking. This is benchmark oracle self-consistency, not an independent visual QA model evaluation.'}
 OUT.write_text(json.dumps(report,indent=2)+'\n')
 print('SADU_ORACLE_SOURCE_AUDIT_DIAGRAMS',len(results),'QA_INVARIANTS',report['numeric_QA_invariants_tested'],'MISMATCHES',len(mismatches),'PER_PROPERTY',summary,'SAVED',OUT)
 if mismatches:print('FIRST_SOURCE_MISMATCHES',mismatches[:14])
if __name__=='__main__':main()
