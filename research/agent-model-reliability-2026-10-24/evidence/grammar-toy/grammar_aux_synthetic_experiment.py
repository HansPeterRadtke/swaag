#!/usr/bin/env python3
"""CPU-only, frozen small-model ablation; cannot establish results for Qwen.

Compare three *same-parameter-count* GRU language models on identical generated
sentences, updates and held-out lemma pairings: LM only, LM+genuine grammatical
role/number tags, LM+shuffled tags. Independent grammar generates targets.
"""
import argparse
import itertools
import json
import math
from pathlib import Path
import random
import statistics
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

torch.set_num_threads(1)
VERSION='2026-10-24-grammar-aux-predeclared-v1'
LEMMAS=['cat','dog','bird','horse','fox']
ACTIONS=['run','jump','sleep']
SPECIAL=['<bos>','the','near','<eos>']
VOCAB=SPECIAL+[x+suffix for x in LEMMAS for suffix in ('','s')]+[x+suffix for x in ACTIONS for suffix in ('','s')]
ID={t:i for i,t in enumerate(VOCAB)}
ROLE_TAGS=['BOS','DET_SUBJ','SUBJ_SG','SUBJ_PL','NEAR','DET_DIST','DIST_SG','DIST_PL','VERB_SG','VERB_PL']
TAG_ID={t:i for i,t in enumerate(ROLE_TAGS)}


def corpus():
    pairs=list(itertools.product(LEMMAS,LEMMAS))
    random.Random(20261024).shuffle(pairs)
    groups={'train':pairs[:16],'validation':pairs[16:20],'test':pairs[20:]}
    def example(subject,distractor,subject_plural,dist_plural,action):
        # Always exactly 8 tokens; role labels are generated from the grammar.
        words=['<bos>','the',subject+('s' if subject_plural else ''),'near','the',
               distractor+('s' if dist_plural else ''),action+('' if subject_plural else 's'),'<eos>']
        roles=['BOS','DET_SUBJ','SUBJ_PL' if subject_plural else 'SUBJ_SG',
               'NEAR','DET_DIST','DIST_PL' if dist_plural else 'DIST_SG',
               'VERB_PL' if subject_plural else 'VERB_SG']
        return ([ID[z] for z in words],[TAG_ID[z] for z in roles],int(subject_plural),subject,distractor)
    result={}
    for split,values in groups.items():
        result[split]=[example(s,d,sp,dp,a) for s,d in values for sp in (False,True)
                  for dp in (False,True) for a in ACTIONS]
    assert set(groups['train']).isdisjoint(groups['test'])
    assert len(result['train'])==192 and len(result['validation'])==48 and len(result['test'])==60
    return result,groups


class TinyGrammarLM(nn.Module):
    def __init__(self,seed):
        super().__init__()
        torch.manual_seed(seed)
        self.embedding=nn.Embedding(len(VOCAB),16)
        self.recurrent=nn.GRU(16,32,batch_first=True)
        self.lm_head=nn.Linear(32,len(VOCAB))
        self.syntax_head=nn.Linear(32,len(ROLE_TAGS))
    def forward(self,tokens):
        states,_=self.recurrent(self.embedding(tokens))
        return self.lm_head(states),self.syntax_head(states)


def tensors(rows):
    return (torch.tensor([row[0] for row in rows],dtype=torch.long),
            torch.tensor([row[1] for row in rows],dtype=torch.long),
            torch.tensor([row[2] for row in rows],dtype=torch.long))


def evaluate(net, rows):
    net.eval()
    words,_,plural=tensors(rows)
    with torch.inference_mode():
        logits,_=net(words[:,:-1]); target=words[:,1:]
        loss=F.cross_entropy(logits.reshape(-1,len(VOCAB)),target.reshape(-1)).item()
        verb_log=logits[:,5].log_softmax(dim=-1)
        plural_ids=torch.tensor([ID[a] for a in ACTIONS])
        singular_ids=torch.tensor([ID[a+'s'] for a in ACTIONS])
        plural_mass=torch.logsumexp(verb_log[:,plural_ids],dim=-1)
        sing_mass=torch.logsumexp(verb_log[:,singular_ids],dim=-1)
        correct=((plural_mass>sing_mass).long()==plural).float().mean().item()
    return {'next_token_cross_entropy':round(loss,5),'verb_number_accuracy':round(correct,5)}


def train_one(mode,seed,data,epochs,batch):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    net=TinyGrammarLM(seed)
    optimizer=torch.optim.AdamW(net.parameters(),lr=0.006,weight_decay=0.0)
    train_words,train_tags,_=tensors(data['train'])
    # Same globally shuffled label histogram and same optimizer-step count.
    shuffled=train_tags.flatten()[torch.randperm(train_tags.numel(),generator=torch.Generator().manual_seed(10000+seed))].reshape_as(train_tags)
    n=len(train_words)
    steps=0
    started=time.perf_counter()
    for ep in range(epochs):
        permutation=torch.randperm(n,generator=torch.Generator().manual_seed(5000+seed*17+ep))
        for offset in range(0,n,batch):
            inds=permutation[offset:offset+batch]
            tokens=train_words[inds,:-1]
            targets=train_words[inds,1:]
            tags=(shuffled if mode=='shuffled' else train_tags)[inds]
            optimizer.zero_grad(set_to_none=True)
            lm_logits,syntax_logits=net(tokens)
            lm_loss=F.cross_entropy(lm_logits.reshape(-1,len(VOCAB)),targets.reshape(-1))
            syn_loss=F.cross_entropy(syntax_logits.reshape(-1,len(ROLE_TAGS)),tags.reshape(-1))
            # Compute auxiliary head in every mode; syntax/shuffled share same
            # exact coefficient, batch count, architecture, train tokens/updates.
            weight=0.0 if mode=='plain' else 0.3
            (lm_loss+weight*syn_loss).backward()
            optimizer.step(); steps+=1
    seconds=time.perf_counter()-started
    return {'mode':mode,'seed':seed,'train_examples':n,'epochs':epochs,'updates':steps,
        'total_tokens_seen':n*epochs*7,'architecture_parameters':sum(z.numel() for z in net.parameters()),
        'train_seconds':round(seconds,3),'validation':evaluate(net,data['validation']),
        'heldout_test':evaluate(net,data['test'])}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=5);p.add_argument('--batch',type=int,default=32)
    a=p.parse_args()
    if (a.epochs,a.batch)!=(5,32):raise SystemExit('Only frozen protocol epochs=5 batch=32 supported')
    if a.output.exists():raise SystemExit('Refusing to overwrite experiment evidence')
    data,groups=corpus()
    rows=[]
    for seed in (11,23,41):
        for mode in ('plain','syntax','shuffled'):
            r=train_one(mode,seed,data,a.epochs,a.batch)
            rows.append(r)
            print(mode,seed,r['heldout_test'],flush=True)
    result={'revision':VERSION,'training_environment':{'torch':torch.__version__,'device':'CPU',
        'available_threads':torch.get_num_threads()},'model':'single-layer GRU 16d embed 32d hidden, LM+aux heads all conditions',
        'corpus':'fixed eight-token generated agreement language, disjoint lemma-pair splits',
        'splits':{name:len(v) for name,v in data.items()},
        'split_pairs':{name:groups[name] for name in ('train','validation','test')},
        'conditions':'plain no aux loss; syntax real generated labels; shuffled random syntax labels, same histogram',
        'optimizer':'AdamW lr 0.006 weight_decay 0','epochs':a.epochs,'batch':a.batch,
        'aux_weight':0.3,'predeclared_seeds':[11,23,41],'results':rows}
    result['summary']={mode:{metric:{'mean':round(statistics.mean(x['heldout_test'][metric] for x in rows if x['mode']==mode),5),
         'values':[x['heldout_test'][metric] for x in rows if x['mode']==mode]} for metric in ('next_token_cross_entropy','verb_number_accuracy')}
         for mode in ('plain','syntax','shuffled')}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['summary'],indent=2),flush=True)

if __name__=='__main__':main()
