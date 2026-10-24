"""Independent exhaustive Boolean entailment; not influenced by model outputs."""
import itertools
import json
from pathlib import Path


def entailment(case):
    atoms=case['atoms']
    if len(set(atoms))!=len(atoms) or not 0<len(atoms)<=12 or case['target'] not in atoms:
        raise ValueError('invalid_atoms')
    if any(name not in atoms or type(truth) is not bool for name,truth in case['observed'].items()):
        raise ValueError('invalid_observation')
    if any(len(r)!=2 or r[0] not in atoms or r[1] not in atoms for r in case['rules']):
        raise ValueError('invalid_implication')
    possible=[]
    for truth_values in itertools.product((False,True),repeat=len(atoms)):
        world=dict(zip(atoms,truth_values))
        if any(world[name]!=truth for name,truth in case['observed'].items()):continue
        if any(world[left] and not world[right] for left,right in case['rules']):continue
        possible.append(world[case['target']])
    if not possible: raise ValueError('contradictory_facts')
    return 'yes' if all(possible) else 'no' if not any(possible) else 'unknown'


def load(path):
    cases=json.loads(Path(path).read_text())
    ids=[c['id'] for c in cases]
    if len(ids)!=len(set(ids)):raise ValueError('duplicate_case')
    return [dict(c,gold=entailment(c)) for c in cases]
