"""Protocol guardrails, no oracle source IDs to agent, honest missing state."""
import random
from pathlib import Path
import numpy as np
import pytest
from experiment import Reader,World,RawPolicy,collect_experience,corpus,sha,score_outcomes,choose_threshold

def tiny():
 return [dict(query='alpha action reference explained',body='Alpha action reference explained with details about the operation and its precise outcomes.',source='PRIVATE',qualified='secret'),dict(query='beta distinct example explained',body='Beta distinct example described in a separate document with characteristics unlike alpha.',source='PRIVATE',qualified='secret')]

def test_world_returns_bit_not_hidden_target():
 r=tiny();w=World(r)
 assert w.act(r[0],0) is True
 assert w.act(r[0],1) is False
 assert type(w.act(dict(query='missing',body='missing'),0)) is bool

def test_source_fields_not_used_by_retriever():
 data=tiny();clean=[x['body'] for x in data]
 a=Reader(data);b=Reader(clean)
 for q in ('alpha reference','beta example'):
  p,_=a.rank(q);r,_=b.rank(q)
  assert [(i,x.tolist()) for i,x in p]==[(i,x.tolist()) for i,x in r]

def test_raw_policy_has_trainable_input_gradients():
 import torch
 model=RawPolicy(True)
 xs=np.ones((2,8),dtype=np.float32)
 loss=model(['alpha reference','beta operation'],[tiny()[0]['body'],tiny()[1]['body']],xs).sum()
 loss.backward()
 assert model.emb.weight.grad is not None
 assert torch.count_nonzero(model.emb.weight.grad)>0

def test_missing_evidence_does_not_fabricate_success():
 docs=tiny();req=[dict(query='gamma not here',body='gamma completely missing')]
 a=score_outcomes([dict(score=.99,chosen=0,chosen_body_hash=sha(docs[0]['body']))],req,docs,0.)
 assert a['correct']==0 and a['wrong']==1 and a['unknown_false']==1

def test_abstain_is_counted_and_costed():
 docs=tiny();req=[dict(query='unknown',body='unknown')]
 a=score_outcomes([dict(score=.5,chosen=0)],req,docs,.7)
 assert a['abstained']==1 and a['correct']==a['wrong']==0
 assert a['net_reward']<0

def test_corpus_disjoint_exhaustive():
 d,excluded=corpus();bs=set();qs=set()
 for k,rs in d.items():
  for r in rs:
   b=sha(r['body']);q=sha(r['query'].casefold())
   assert b not in bs and q not in qs
   bs.add(b);qs.add(q)
 assert len(d['test_known'])>=100 and len(d['test_unknown'])>=15

def test_feedback_contains_only_chosen_text_and_reward():
 docs=tiny();ev,info=collect_experience(docs,1701,subsets=2)
 assert ev
 assert all(len(e)==6 and isinstance(e[3],int) and e[3] in (0,1) for e in ev)
 assert all(e[0] in [x['query'] for x in docs] for e in ev)
 assert info['actions']==len(ev)

def test_calibration_min_coverage_constraint():
 docs=tiny();p=[dict(score=.9,chosen=0),dict(score=.2,chosen=1)]
 threshold,cal=choose_threshold(p,docs,docs,min_coverage=.5)
 assert cal['dev_accepted']/cal['n']>=.5
