"""COG-ATTN-097e: outcome-trained, attention-free retrieval/action/abstain agent.

Input to learner: raw naturally authored text and an index of raw archived bodies.
Learning signal: ONLY one-bit reward for independently chosen retrieval actions.
Never passed to learner: gold source ID, entity ID, field tags, semantic matches,
pre-selected correct candidates, absent/known flags, test-time feedback.

Intended scientific claim: bounded retrieval subsystem diagnostic, *not* integrated
COG-ATTN cognition or human task-level intelligence.
"""
from __future__ import annotations
import argparse, collections, hashlib, importlib.util, json, math, random, statistics, time
from pathlib import Path
from functools import lru_cache
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import torch
from torch import nn
from torch.nn import functional as F
from vendor_097c_experiment import make_corpus

TRAIN_MODULES=('argparse','subprocess','collections','typing','http.client',
  'email.message','unittest.case','asyncio.base_events','urllib.request',
  'logging','inspect','tarfile','zipfile','json.encoder','contextlib',
  'tempfile','doctest','threading','configparser','platform','email.utils',
  'base64','functools','dataclasses','uuid','random')
DEV_KNOWN_MODULES=('statistics','unittest.mock','socket')
DEV_UNKNOWN_MODULES=('pkgutil','trace','copy','html.parser','email.feedparser','socketserver','xml.dom.minidom')
TEST_KNOWN_MODULES=('smtplib','imaplib','difflib','cmd','queue','calendar','ipaddress',
 'ftplib','logging.handlers','pickletools','fileinput','xml.sax.handler',
 'xml.sax.xmlreader','tokenize','codeop','code','numbers','shelve',
 'wsgiref.handlers','email.mime.text','pprint','enum','ast','unittest.suite')
TEST_UNKNOWN_MODULES=('shutil','pickle','json.decoder','textwrap')
SEEDS=(1701,1707,1711)

V=16384
D=32

def sha(s):return hashlib.sha256(s.encode('utf-8')).hexdigest()
@lru_cache(maxsize=16000)
def tokenize(s):
    import re
    return re.findall(r"[a-zA-Z_][a-zA-Z_0-9]*",s.lower())[:120]
@lru_cache(maxsize=16000)
def ids(s):
    ts=tokenize(s); z=[]
    for t in ts:
        z.append(int.from_bytes(hashlib.blake2s(('w'+t).encode(),digest_size=4).digest(),'little')%V)
        decorated='^'+t+'$'
        for j in range(len(decorated)-2):
            z.append(int.from_bytes(hashlib.blake2s(('c'+decorated[j:j+3]).encode(),digest_size=4).digest(),'little')%V)
    return z[:480] or [0]

def corpus():
    spec=[('train',TRAIN_MODULES),('dev_known',DEV_KNOWN_MODULES),
          ('dev_unknown',DEV_UNKNOWN_MODULES),('test_known',TEST_KNOWN_MODULES),
          ('test_unknown',TEST_UNKNOWN_MODULES)]
    seenb=set();seenq=set();data={};excluded={}
    for split,modules in spec:
        raw=make_corpus(modules);data[split]=[];excluded[split]=0
        for r in raw:
            b=sha(r['body']);q=sha(r['query'].casefold())
            if b in seenb or q in seenq:
                excluded[split]+=1;continue
            data[split].append(r);seenb.add(b);seenq.add(q)
    assert all(data.values())
    assert len(set(sum((list(m) for _,m in spec),[])))==sum(len(m) for _,m in spec)
    return data,excluded

class Reader:
    def __init__(self,archive):
        # Learner sees only raw bodies; no source identity/qualified names.
        self.archive=list(archive)
        self.bodies=[d['body'] if isinstance(d,dict) else d for d in archive]
        self.wv=TfidfVectorizer(analyzer='word',ngram_range=(1,2),sublinear_tf=True)
        self.cv=TfidfVectorizer(analyzer='char',ngram_range=(3,5),sublinear_tf=True)
        t=time.perf_counter_ns()
        self.wmat=self.wv.fit_transform(self.bodies)
        self.cmat=self.cv.fit_transform(self.bodies)
        self.build_ns=time.perf_counter_ns()-t

    def rank(self,q,k=8):
        t=time.perf_counter_ns()
        w=(self.wv.transform([q])@self.wmat.T).toarray()[0]
        c=(self.cv.transform([q])@self.cmat.T).toarray()[0]
        wordorder=np.argsort(-w,kind='stable')[:k]
        charorder=np.argsort(-c,kind='stable')[:k]
        # Global whole-archive lexical retrieval, no gold or task-specific slots.
        choices=list(dict.fromkeys([int(i) for i in charorder]+[int(i) for i in wordorder]))
        candidates=[]
        c_second=float(np.partition(c,-2)[-2]) if len(c)>1 else 0.0
        w_second=float(np.partition(w,-2)[-2]) if len(w)>1 else 0.0
        for j in choices:
            # Additional scalar features are computed from visible text only.
            toks=set(tokenize(q)); dtoks=set(tokenize(self.bodies[j]))
            overlap=len(toks&dtoks)/(len(toks) or 1)
            candidates.append((j,np.array([float(c[j]),float(w[j]),overlap,
                float(c[j]-c_second),
                float(w[j]-w_second),
                math.log1p(len(tokenize(q))),math.log1p(len(tokenize(self.bodies[j]))),
                1.0],dtype=np.float32)))
        return candidates,time.perf_counter_ns()-t

class World:
    """Not imported by learner. The model gets ONLY bool from act()."""
    def __init__(self,archive):
        self.hidden={sha(r['body']):i for i,r in enumerate(archive)}
        self.calls=0
    def act(self,request,chosen):
        self.calls+=1
        target=self.hidden.get(sha(request['body']))
        # No target, missing status, source metadata, or correct action escapes.
        return bool(chosen==target)

class RawPolicy(nn.Module):
    """End-to-end trainable perception + action utility; NO attention."""
    def __init__(self,with_encoder=True):
        super().__init__();self.with_encoder=with_encoder
        if with_encoder:
            self.emb=nn.EmbeddingBag(V,D,mode='mean')
            self.head=nn.Sequential(nn.Linear(D*3+8,64),nn.GELU(),nn.Linear(64,1))
        else:
            self.head=nn.Sequential(nn.Linear(8,24),nn.GELU(),nn.Linear(24,1))
    def embed(self,lines):
        z=[ids(t) for t in lines]; lengths=[len(x) for x in z]
        flat=torch.tensor([i for zz in z for i in zz],dtype=torch.long)
        offsets=torch.tensor([0]+list(np.cumsum(lengths[:-1])),dtype=torch.long)
        return self.emb(flat,offsets)
    def forward(self,qs,ds,xs):
        xs=torch.as_tensor(np.asarray(xs,dtype=np.float32))
        if self.with_encoder:
            q=self.embed(qs);d=self.embed(ds)
            v=torch.cat([q,d,torch.abs(q-d),xs],dim=1)
        else:v=xs
        return self.head(v).flatten()


def collect_experience(rows,seed,subsets=3):
    # No scorer exposes which candidate is right until a chosen ACTION occurs.
    # Construct naturally incomplete episodic histories to teach uncertainty.
    rng=random.Random(seed*991);events=[];counts=collections.Counter();t=time.perf_counter_ns()
    for s in range(subsets):
        positions=list(range(len(rows)));rng.shuffle(positions)
        # Some docs missing unpredictably, so both answerable and missing trials occur.
        retain=set(positions[:int(len(rows)*(.5+.075*s))])
        archive=[rows[i] for i in positions if i in retain]
        reader=Reader(archive);world=World(archive)
        order=list(range(len(rows)));rng.shuffle(order)
        for rix in order:
            req=rows[rix];candidates,_=reader.rank(req['query'],k=6)
            # Outcome-driven exploratory actions from retrieval, not a gold list.
            # Strictly choose from ranking of observed text; 3 action budget.
            seen=set();order_char=sorted(candidates,key=lambda v:(-v[1][0],v[0]));
            order_word=sorted(candidates,key=lambda v:(-v[1][1],v[0]));
            proposals=[]
            for jj in (order_char[:2]+order_word[:2]):
                if jj[0] not in seen:seen.add(jj[0]);proposals.append(jj)
            for j,x in proposals[:3]:
                reward=world.act(req,j)
                events.append((req['query'],reader.bodies[j],x,int(reward),sha(req['query']),sha(reader.bodies[j])))
                counts['actions']+=1;counts['successes']+=reward
                if reward:break
        counts['index_ns']+=reader.build_ns
    counts['episodes']=len(rows)*subsets
    counts['experience_collection_ns']=time.perf_counter_ns()-t
    return events,dict(counts)


def train_policy(events,seed,with_encoder=True,shuffle=False,epochs=3):
    torch.manual_seed(seed+333);torch.set_num_threads(1)
    model=RawPolicy(with_encoder)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    yy=[x[3] for x in events]; rng=random.Random(seed+99)
    if shuffle:rng.shuffle(yy)
    npos=sum(yy);pos_weight=(len(yy)-npos)/max(1,npos)
    curve=[];updates=0;t=time.perf_counter_ns()
    for ep in range(epochs):
        idx=list(range(len(events)));random.Random(seed+ep*233).shuffle(idx)
        losses=[]
        for j in range(0,len(idx),96):
            inds=idx[j:j+96]
            if len(inds)<8:continue
            logits=model([events[i][0] for i in inds],
                [events[i][1] for i in inds],[events[i][2] for i in inds])
            y=torch.tensor([yy[i] for i in inds],dtype=torch.float32)
            loss=F.binary_cross_entropy_with_logits(logits,y,
                        pos_weight=torch.tensor(float(pos_weight)))
            opt.zero_grad();loss.backward();opt.step()
            updates+=1;losses.append(float(loss.detach()))
        curve.append(round(float(np.mean(losses)),6))
    model.eval()
    return model,dict(loss=curve,updates=updates,elapsed_ns=time.perf_counter_ns()-t,
        positives=npos,negatives=len(yy)-npos,shuffled=shuffle)

@torch.inference_mode()
def score_policy(model,requests,archive,mode='neural'):
    reader=Reader(archive);t0=time.perf_counter_ns();rows=[]
    for r in requests:
        candidates,_=reader.rank(r['query'],k=8)
        if mode=='char':
            j,x=max(candidates,key=lambda z:(z[1][0],-z[0]));value=float(x[0]);
        elif mode=='word':
            j,x=max(candidates,key=lambda z:(z[1][1],-z[0]));value=float(x[1]);
        else:
            logits=model([r['query']]*len(candidates),[reader.bodies[j] for j,_ in candidates],[x for _,x in candidates])
            probs=torch.sigmoid(logits).numpy()
            order=np.argsort(-probs,kind='stable')
            ix=int(order[0]);j,x=candidates[ix];value=float(probs[ix])
        rows.append(dict(query_hash=sha(r['query']),chosen_body_hash=sha(reader.bodies[j]),
            score=value,features=x.tolist(),chosen=int(j)))
    return rows,dict(build_ns=reader.build_ns,online_ns=time.perf_counter_ns()-t0)

def score_outcomes(rows,requests,archive,threshold):
    # **Only evaluator** can inspect source association.
    world=World(archive);correct=wrong=abstained=unknown_false=0
    records=[]
    for pred,req in zip(rows,requests):
        choose=pred['score']>=threshold
        target_present=sha(req['body']) in world.hidden
        if choose:
            ok=world.act(req,pred['chosen']);correct+=ok;wrong+=not ok
            if not target_present:unknown_false+=1
        else:abstained+=1;ok=False
        records.append(dict(request=sha(req['query']),present=target_present,score=pred['score'],
            chosen=pred['chosen_body_hash'] if choose else None,reward=int(ok),
            deferred=not choose))
    accepted=correct+wrong
    return dict(n=len(rows),correct=correct,wrong=wrong,abstained=abstained,
      accepted=accepted,precision=correct/accepted if accepted else 0,
      coverage=accepted/len(rows),unknown_false=unknown_false,
      net_reward=correct-2*wrong-.05*abstained,
      records=records)

def choose_threshold(rows,requests,archive,min_coverage=.15):
    # Calibration only from outcomes of agent-selected actions on disjoint DEV.
    # Evaluator returns success bit for selected action; no target index to model.
    world=World(archive)
    ordered=[]
    for p,q in zip(rows,requests):
        ordered.append((p['score'],int(world.act(q,p['chosen']))))
    candidates=sorted({float(p['score']) for p in rows}|{float('inf'),-float('inf')})
    best=(-1e12,None,None)
    for threshold in candidates:
        accept=[int(score>=threshold) for score,_ in ordered]
        if sum(accept)/len(ordered)<min_coverage:continue
        correct=sum(a*r for a,(_,r) in zip(accept,ordered))
        wrong=sum(a*(1-r) for a,(_,r) in zip(accept,ordered))
        utility=correct-2*wrong-.05*(len(rows)-sum(accept))
        score=utility
        if score>best[0]:best=(score,threshold,sum(accept))
    # Threshold selection restricted by dev coverage; no touching TEST.
    return float(best[1]),dict(dev_utility=best[0],dev_accepted=best[2],n=len(rows),
                               coverage_floor=min_coverage,dev_actions=len(ordered))

def run(seed=1701,out=None,weights=None,epochs=3):
    torch.set_num_threads(1)
    datasets,excluded=corpus();rng=random.Random(seed)
    # Different archives and unseen queries for development vs test.
    dev_archive=list(datasets['dev_known']);rng.shuffle(dev_archive)
    test_archive=list(datasets['test_known']);rng.shuffle(test_archive)
    dev_req=list(datasets['dev_known'])+list(datasets['dev_unknown']);rng.shuffle(dev_req)
    test_req=list(datasets['test_known'])+list(datasets['test_unknown']);rng.shuffle(test_req)
    events,experience=collect_experience(datasets['train'],seed)
    assert any(e[3] for e in events) and any(not e[3] for e in events)
    models={};train_info={}
    for name,encoder,shuf in [('learned',True,False),('no_encoder',False,False),('shuffled_rewards',True,True)]:
        m,receipt=train_policy(events,seed,with_encoder=encoder,shuffle=shuf,epochs=epochs)
        models[name]=m;train_info[name]=receipt
    if weights:
        path=Path(weights);path.mkdir(parents=True,exist_ok=True)
        torch.save(dict(seed=seed,state=models['learned'].state_dict(),
            events_digest=sha('\n'.join(f'{e[4]}|{e[5]}|{e[3]}' for e in events))),path/f'097e-{seed}.pt')
    predictions={};results={};time_info={};calibration={}
    for name in ('word','char','learned','no_encoder','shuffled_rewards'):
        mode=name if name in ('char','word') else 'neural'
        model=models.get(name)
        dev_pred,dev_times=score_policy(model,dev_req,dev_archive,mode)
        cut,info=choose_threshold(dev_pred,dev_req,dev_archive)
        test_pred,times=score_policy(model,test_req,test_archive,mode)
        result=score_outcomes(test_pred,test_req,test_archive,cut)
        # Separate always-answer accuracy for fair paired comparison.
        always=score_outcomes(test_pred,test_req,test_archive,-float('inf'))
        result['always_correct']=always['correct'];result['always_n']=always['n']
        result['known_n']=sum(sha(x['body']) in {sha(z['body']) for z in test_archive} for x in test_req)
        result['unknown_n']=len(test_req)-result['known_n']
        predictions[name]=test_pred
        calibration[name]=dict(threshold=cut,**info)
        time_info[name]=dict(dev=dev_times,test=times)
        results[name]=result
    # No source/target ID in the training events; test only after frozen optimization/calibration.
    meta=dict(seeds=[seed],status='097e raw-text outcome-trained retrieval-action subsystem, NOT full COG-ATTN',
         python_version=__import__('sys').version,
         data_counts={k:len(x) for k,x in datasets.items()},duplicates_removed=excluded,
         protocol='no gold identifier, answer list, relation inventory, parse schema or query->passage teaching; only binary reward for executed chosen action; archive incomplete in training',
         fresh_test_no_feedback=True,
         event_receipt=experience,train=train_info,calibration=calibration,results=results,
         times=time_info,
         test_input_sha=sha('\n'.join(x['query'] for x in test_req)),
         train_input_sha=sha('\n'.join(x['body'] for x in datasets['train'])),
         test_modules=TEST_KNOWN_MODULES,test_unknown_modules=TEST_UNKNOWN_MODULES,
         source_sha={m:sha(Path(importlib.util.find_spec(m).origin).read_text(encoding='utf-8'))
            for m in TRAIN_MODULES+DEV_KNOWN_MODULES+DEV_UNKNOWN_MODULES+TEST_KNOWN_MODULES+TEST_UNKNOWN_MODULES if importlib.util.find_spec(m).origin and importlib.util.find_spec(m).origin.endswith('.py')},
         feedback_trace=[dict(query=e[4],proposed=e[5],reward=e[3]) for e in events],
         caveats=['Constructed retrieval decisions from naturally authored Python docstrings, not human dialogues',
          'Source association is held by simulator; the 0/1 action reward is synthetic, not a real user',
          'Generic lexical candidate retrieval is part of the learner and must be charged, not oracle selection',
          'No user-authenticated execution, autonomous skill formation or full COG-ATTN controller',
          'Model did not train on test query/document pairs; developmental calibration uses only disjoint modules',
          'All seeds reuse same held-out documents, so per-seed evaluations are correlated',
          'Trial uses reward on proposed document ID only and optional ABSTAIN, not natural-language response'])
    if out:Path(out).write_text(json.dumps(meta,indent=2,sort_keys=True)+'\n')
    return meta

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=1701)
    ap.add_argument('--epochs',type=int,default=6)
    ap.add_argument('--out',required=True);ap.add_argument('--weights',default=None)
    ar=ap.parse_args();r=run(ar.seed,ar.out,ar.weights,ar.epochs)
    print(json.dumps({"seed":ar.seed,"counts":r['data_counts'],
        "events":r['event_receipt'],
        "results":{k:{f:v[f] for f in ('correct','wrong','abstained','precision','coverage','unknown_false','always_correct')}
        for k,v in r['results'].items()}},indent=2))
