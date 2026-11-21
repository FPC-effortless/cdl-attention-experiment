"""COG-ATTN-097d: attention-free self-supervised experience learning.

Learner input: raw authored documentation BODY TEXT only, contiguous windows
from the same observed document stream. Learner never receives query–document
links, source-module ids, author summaries, action labels or sealed test data.
Training feedback is the *next observed raw window*, not an oracle label.
Evaluation's hidden source association is used only by the independent scorer.

This is a trainable perceptual retrieval subsystem, NOT the full COG agent.
"""
from __future__ import annotations
import argparse, collections, hashlib, importlib.util, json, math, os, random, re, statistics, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.feature_extraction.text import TfidfVectorizer

from vendor_097c_experiment import make_corpus

# Freeze by *source module*, not randomly splitting snippets of one source.
TRAIN_MODULES = ('argparse','subprocess','collections','typing','http.client',
  'email.message','unittest.case','asyncio.base_events','urllib.request',
  'logging','inspect','tarfile','zipfile','json.encoder','contextlib',
  'tempfile','doctest','threading','configparser','platform','email.utils',
  'base64','functools','dataclasses','uuid','random')
DEV_MODULES = ('statistics','unittest.mock','socket')
TEST_MODULES = ('xml.etree.ElementTree','traceback','email.header',
  'email.parser','email.generator','fractions','weakref','re','glob','types','pydoc')
UNKNOWN_MODULES = ('shutil','pickle','json.decoder','textwrap')
SEEDS=(1701,1707,1711)
WORD=re.compile(r"[a-zA-Z_][a-zA-Z_0-9]*")
VOCAB=32768
DIM=64

def sha(s):return hashlib.sha256(s.encode()).hexdigest()
def h(s):return int.from_bytes(hashlib.blake2s(s.encode('utf-8'),digest_size=4).digest(),'little')%VOCAB

def features(s):
    words=WORD.findall(s.lower())
    arr=[]
    for w in words[:240]:
        arr.append(h('w:'+w))
        decorated='^'+w+'$'
        for j in range(max(0,len(decorated)-2)):
            arr.append(h('c:'+decorated[j:j+3]))
    # order is no longer represented; a future model may need it.
    return arr[:1400] or [h('<empty>')]

def pair_windows(body):
    words=body.split()
    # The 097c extraction minimum is 28 words, so two disjoint windows exist.
    cut=len(words)//2
    return ' '.join(words[:cut]), ' '.join(words[cut:])

class Encoder(nn.Module):
    """Trainable non-attention word/subword memory encoder with pooled features."""
    def __init__(self):
        super().__init__()
        self.table=nn.EmbeddingBag(VOCAB,DIM,mode='mean')
        self.ff=nn.Sequential(nn.Linear(DIM,96),nn.GELU(),nn.Linear(96,DIM))
    def forward(self,contents):
        encoded=[features(x) for x in contents]
        lengths=[len(x) for x in encoded]
        flat=torch.tensor([i for z in encoded for i in z],dtype=torch.long)
        offsets=torch.tensor([0]+list(np.cumsum(lengths[:-1])),dtype=torch.long)
        return F.normalize(self.ff(self.table(flat,offsets)),dim=-1)

def train_model(bodies,seed,epochs=8,mode='adjacent'):
    torch.manual_seed(seed);random.seed(seed)
    net=Encoder();opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.001)
    pairs=[pair_windows(x) for x in bodies]
    start=time.perf_counter_ns();curve=[];updates=0
    for ep in range(epochs):
        perm=list(range(len(pairs)));random.Random(seed+ep*147).shuffle(perm)
        losses=[]
        for k in range(0,len(perm),24):
            chosen=perm[k:k+24]
            if len(chosen)<4:continue
            a=[pairs[i][0] for i in chosen]
            b=[pairs[i][1] for i in chosen]
            if mode=='shuffled': b=b[1:]+b[:1]
            qa=net(a);qb=net(b)
            logits=(qa @ qb.T)/.12
            labels=torch.arange(len(chosen))
            loss=(F.cross_entropy(logits,labels)+F.cross_entropy(logits.T,labels))/2
            opt.zero_grad();loss.backward();opt.step()
            updates+=1;losses.append(float(loss.detach()))
        curve.append(round(statistics.mean(losses),5))
    return net, dict(loss=curve,updates=updates,train_ns=time.perf_counter_ns()-start)

@torch.inference_mode()
def embed(net,strings,batch=48):
    net.eval();return torch.cat([net(strings[i:i+batch]) for i in range(0,len(strings),batch)]).numpy()

def rank_embedding(test,net):
    bodies=[x['body'] for x in test];queries=[x['query'] for x in test]
    start=time.perf_counter_ns()
    b=embed(net,bodies)
    build_ns=time.perf_counter_ns()-start
    start=time.perf_counter_ns();q=embed(net,queries)
    scores=q @ b.T
    elapsed_ns=time.perf_counter_ns()-start
    return scores, build_ns, elapsed_ns

def rank_tfidf(test,kind):
    bodies=[x['body'] for x in test];queries=[x['query'] for x in test]
    start=time.perf_counter_ns();v=TfidfVectorizer(analyzer=kind,
      ngram_range=(1,2) if kind=='word' else (3,5),sublinear_tf=True)
    b=v.fit_transform(bodies);build_ns=time.perf_counter_ns()-start
    start=time.perf_counter_ns();scores=(v.transform(queries) @ b.T).toarray()
    elapsed_ns=time.perf_counter_ns()-start
    return scores,build_ns,elapsed_ns

def scoring(scores):
    n=len(scores)
    top=np.argsort(-scores,axis=1,kind='stable')
    rank=np.array([int(np.where(top[i]==i)[0][0])+1 for i in range(n)])
    return dict(correct=int((rank==1).sum()),n=n,
      accuracy=round(float((rank==1).mean()),6),
      recall_at_5=round(float((rank<=5).mean()),6),
      mrr=round(float(np.mean(1.0/rank)),6),
      per_query_rank=[int(x) for x in rank])

def run(seed=1701,epochs=8,out=None,checkpoint=None):
    # Source module labels are used by test harness only for grouping and never
    # passed into any of the trainable encoders or lexical readers.
    datasets={name:make_corpus(modules) for name,modules in
              [('train',TRAIN_MODULES),('dev',DEV_MODULES),('test',TEST_MODULES),
               ('unknown',UNKNOWN_MODULES)]}
    assert all(datasets.values())
    # Deduplicate across splits using ORIGINAL prose and author summaries.
    # Identical inspect.getdoc / pydoc._getdoc docs cross source modules.
    # Priority train -> dev -> test -> unknown fixed before final re-scoring.
    seen_body=set();seen_query=set();excluded={}
    for split in ('train','dev','test','unknown'):
        retained=[];excluded[split]=0
        for item in datasets[split]:
            bh=sha(item['body']);qh=sha(item['query'].casefold())
            if bh in seen_body or qh in seen_query:
                excluded[split]+=1;continue
            retained.append(item);seen_body.add(bh);seen_query.add(qh)
        datasets[split]=retained
    assert all(datasets.values())
    assert not (set(TRAIN_MODULES)&set(TEST_MODULES))
    assert not (set(DEV_MODULES)&set(TEST_MODULES))
    assert not (set(UNKNOWN_MODULES)&set(TEST_MODULES))
    train=datasets['train'];test=datasets['test'];unknown=datasets['unknown']
    # Training never observes author summary queries, even for training modules.
    torch.set_num_threads(1)
    models={}
    torch.manual_seed(seed);models['untrained']=Encoder()
    trained,receipt=train_model([r['body'] for r in train],seed,epochs,'adjacent')
    models['self_supervised']=trained
    shuffled,shuf_receipt=train_model([r['body'] for r in train],seed,epochs,'shuffled')
    models['mismatched_experiences']=shuffled
    if checkpoint:
        torch.save(dict(state=trained.state_dict(),seed=seed,epochs=epochs,
                        training_data_sha256=sha('\n'.join(r['body'] for r in train))),checkpoint)
    scores={};stats={};score_matrices={}
    for name,kind in [('word_tfidf','word'),('char_tfidf','char')]:
        val,bt,qt=rank_tfidf(test,kind)
        scores[name]=scoring(val);stats[name]=dict(build_ns=bt,query_ns=qt)
        score_matrices[name]=val
    for name,net in models.items():
        val,bt,qt=rank_embedding(test,net)
        scores[name]=scoring(val);stats[name]=dict(build_ns=bt,query_ns=qt)
        score_matrices[name]=val
    # Natural absent-memory queries are from disjoint modules. The current
    # systems are top-1-only, so they necessarily answer every absent case.
    for name in scores:
        scores[name]['false_answers_unknown']=len(unknown)
    result=dict(status='trained raw-experience retrieval diagnostic; NOT integrated COG agent',
      seed=seed,epochs=epochs,counts={k:len(v) for k,v in datasets.items()},
      cross_split_duplicates_excluded=excluded,
      training_observations='disjoint adjacent raw text windows; no summary queries, labels or source IDs to learner',
      model='EmbeddingBag pooled character/word hashed features + MLP; no self-attention',
      validation='source-disjoint evaluation; test used only after frozen training',
      split_modules={k:list(mods) for k,mods in [('train',TRAIN_MODULES),('dev',DEV_MODULES),
        ('test',TEST_MODULES),('unknown',UNKNOWN_MODULES)]},
      source_sha256={mod:hashlib.sha256(Path(importlib.util.find_spec(mod).origin).read_bytes()).hexdigest()
        for mod in TRAIN_MODULES+DEV_MODULES+TEST_MODULES+UNKNOWN_MODULES},
      body_sha256={k:sha('\n'.join(x['body'] for x in rows)) for k,rows in datasets.items()},
      training=receipt,shuffled_training=shuf_receipt,results=scores,timing=stats,
      no_training_leak_tested=not set(sha(x['body']) for x in train)&set(sha(x['body']) for x in test),
      test_queries_sha256=[sha(r['query']) for r in test],
      caveats=['Training uses naturally authored documentation rather than interactive dialogue',
      'Adjacent positive pairs derive from document contiguity, not downstream task feedback',
      'The source-associated first-summary retrieval benchmark is a proxy, not learned language skill',
      'Evaluation returns a document identifier; no natural language answer generation or action execution',
      'The memory is a new trainable read-only subsystem; full COG controller is not integrated',
      'Unseen summaries are author-written first lines from the same documents as test bodies',
      'All top-1-only readers produce false answers on absent documents',
      'No task-specific supervised labels, gold pairs, or schemas supplied during fitting'])
    if out:Path(out).write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=1701)
    ap.add_argument('--epochs',type=int,default=8)
    ap.add_argument('--out',default='result.json');ap.add_argument('--checkpoint',default=None)
    args=ap.parse_args()
    o=run(args.seed,args.epochs,args.out,args.checkpoint)
    print(json.dumps(dict(counts=o['counts'],train=o['training'],scores={n:{k:r[k] for k in ['correct','n','accuracy','recall_at_5','mrr']} for n,r in o['results'].items()}),indent=2))
