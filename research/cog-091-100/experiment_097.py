"""COG-ATTN-097a: episodic grounding with frozen boundaries, synthetic pilot.

A controlled *subsystem* experiment. Reuses real 091-095 model definitions and
training recipes; does NOT invoke the signed COG execution ledger. Results are
not proof of natural-language retrieval, zero-schema induction, or speedup
against matched-quality Transformer deployments.

Run in the 091-100 package directory:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiment_097.py --out 097-results.json
"""
from __future__ import annotations
import argparse, hashlib, json, random, statistics
from collections import defaultdict
from time import perf_counter_ns
import torch
from torch import nn
from semantic import ACTIONS, Router, make_data, to_tensor, vectorize, lex_first
from hypotheses import HypothesisNet, score as hypothesis_score, targets as hypothesis_targets

SEEDS=(9107,9203,9301,9409,9503,9601)
SIZES=(1536,6144,24576)
BUDGETS=(1,2,4,6)
STOP=('please','do','not','actually','choose','rather','than','use','perhaps','or','instead', 'for','regarding','the','note')


def sha(txt):return hashlib.sha256(txt.encode()).hexdigest()


def train_router(seed, mode, epochs=38):
    torch.manual_seed(seed);d=make_data(seed);cache={}
    tr=to_tensor(d['train'],cache);dv=to_tensor(d['dev'],cache)
    net=Router(mode);optimizer=torch.optim.AdamW(net.parameters(),lr=.007,weight_decay=.0001)
    selected=None;best=-1
    start=perf_counter_ns()
    for epoch in range(epochs):
        net.train();perm=torch.randperm(len(tr[2]))
        for ids in perm.split(48):
            loss=nn.functional.cross_entropy(net(tr[0][ids],tr[1][ids]),tr[2][ids]);optimizer.zero_grad();loss.backward();optimizer.step()
        net.eval()
        with torch.no_grad():accuracy=(net(dv[0],dv[1]).argmax(1)==dv[2]).float().mean().item()
        if accuracy>best+1e-8:
            best=accuracy;selected=(epoch+1,{k:v.detach().clone() for k,v in net.state_dict().items()})
    net.load_state_dict(selected[1]);net.eval()
    return net,dict(dev_accuracy=best,chosen_epoch=selected[0],train_ms=(perf_counter_ns()-start)/1e6,
                    parameter_count=sum(p.numel() for p in net.parameters()))


def train_multi(seed,epochs=38):
    torch.manual_seed(seed);d=make_data(seed);cache={}
    tr=to_tensor(d['train'],cache);dv=to_tensor(d['dev'],cache)
    labels=hypothesis_targets(d['train']);net=HypothesisNet()
    optimizer=torch.optim.AdamW(net.parameters(),lr=.007,weight_decay=.0001)
    best=-1;selected=None;start=perf_counter_ns()
    for epoch in range(epochs):
        net.train();perm=torch.randperm(len(tr[2]))
        for ids in perm.split(48):
            logits, candidates=net(tr[0][ids],tr[1][ids]);loss=(nn.functional.cross_entropy(logits,tr[2][ids])
                 +0.5*nn.functional.binary_cross_entropy_with_logits(candidates,labels[ids]))
            optimizer.zero_grad();loss.backward();optimizer.step()
        dscore=hypothesis_score(net,d['dev'],dv)
        current=dscore['correct']/dscore['n'] + dscore['pair_top2']/dscore['ambiguous_n']
        if current>best+1e-9:selected=(epoch+1,{k:v.detach().clone() for k,v in net.state_dict().items()});best=current
    net.load_state_dict(selected[1]);net.eval()
    return net,dict(dev_composite=best,chosen_epoch=selected[0],train_ms=(perf_counter_ns()-start)/1e6,
                    parameter_count=sum(p.numel() for p in net.parameters()))


def make_store(n,seed):
    """N append-only raw records: six distinct relations per opaque entity.

    A query always names an entity explicitly. The action word can be inflected
    or contextually negated. This is *not* a test of learned entity coreference.
    """
    assert n%6==0
    rng=random.Random(seed^n^0x2197);rows=[];names=[]
    for i in range(n//6):
        name=f'ticket{seed}x{i:05d}';names.append(name)
        for a,action in enumerate(ACTIONS):
            detail=f'p{rng.randrange(100000000):08d}'
            raw=f'archive {name} stage {action} entry {detail}'
            rows.append(dict(entity=name,action=a,payload=detail,raw=raw,source_epoch=1,provenance='signed-simulated'))
    # Shuffle ingestion order to break last-slot or first-record shortcuts.
    rng.shuffle(rows)
    return rows,names


class EpisodeIndex:
    def __init__(self,rows):
        start=perf_counter_ns();self.rows=rows;self.index=defaultdict(list)
        self.last_slot={};self.typed={};self.postings=defaultdict(set)
        for i,r in enumerate(rows):
            self.index[r['entity']].append(i)
            self.last_slot[r['entity']]=i
            self.typed[(r['entity'],r['action'])]=i
            for token in set(r['raw'].split()):self.postings[token].add(i)
        self.build_ns=perf_counter_ns()-start
        self.source_epoch={r['entity']:1 for r in rows}

    def retrieve(self,entity,ranking,k,epoch=1):
        # Epoch guards make old candidates unavailable, never silently accepted.
        if entity not in self.index or self.source_epoch[entity]!=epoch:return (),0
        attempts=0;found=[]
        # Lookup work counts only actually visited candidate records.
        for action in ranking[:k]:
            for idx in self.index[entity]:
                attempts+=1
                row=self.rows[idx]
                if row['action']==action and row['source_epoch']==epoch:
                    found.append(idx);break
        return tuple(found),attempts

    def typed_read(self,entity,action):
        idx=self.typed.get((entity,action));return self.rows[idx]['payload'] if idx is not None else None

    def scan(self,entity,action):
        inspected=0
        for row in self.rows:
            inspected+=1
            if row['entity']==entity and row['action']==action:
                return row['payload'],inspected
        return None,inspected

    def append(self,row):
        start=perf_counter_ns();i=len(self.rows)
        self.rows.append(row)
        self.index[row['entity']].append(i)
        self.last_slot[row['entity']]=i
        self.typed[(row['entity'],row['action'])]=i
        for token in set(row['raw'].split()):self.postings[token].add(i)
        return perf_counter_ns()-start

    def invalidate(self,entity):self.source_epoch[entity]+=1


def tensor_one(text,cache):
    vec,length=vectorize(text,cache)
    return torch.tensor([vec],dtype=torch.float32),torch.tensor([length],dtype=torch.long)


def timed_predictions(model,raw_rows):
    cache={};probs=[];latencies=[]
    model.eval()
    with torch.inference_mode():
        for row in raw_rows:
            start=perf_counter_ns();x,l=tensor_one(row['text'],cache)
            value=model(x,l)
            if isinstance(value,tuple):
                action,candidate=value
                probs.append((nn.functional.softmax(action,dim=1)[0].tolist(),torch.sigmoid(candidate)[0].tolist()))
            else:probs.append((nn.functional.softmax(value,dim=1)[0].tolist(),None))
            latencies.append(perf_counter_ns()-start)
    return probs,latencies


def split_queries(seed,store,names):
    rows=make_data(seed)['test'];rng=random.Random(seed^0x9913)
    sampled=rng.sample(names,len(rows))
    return [dict(text=r['text'],entity=sampled[i],gold=r['answer'],
                 pair=r['pair'],novelty=r['novelty'],text_sha256=sha(r['text'])) for i,r in enumerate(rows)]


def evaluate(seed,n,models):
    store,names=make_store(n,seed);before=perf_counter_ns();idx=EpisodeIndex(store)
    build_ns=perf_counter_ns()-before;queries=split_queries(seed,store,names)
    # The identical query string is used to predict semantics; exact entity ID is
    # supplied as external observation, not learned from the text.
    outputs={};per_case=[];time_charge={}
    controls=defaultdict(int)
    for row in queries:
        gold=row['gold']
        if gold==6:continue
        controls['unambiguous']+=1
        lexical=lex_first(row['text'])
        controls['first_verb_action_correct']+=int(lexical==gold)
        controls['eager_slot_contains_answer']+=int(store[idx.last_slot[row['entity']]]['action']==gold)
    for label,model in models.items():
        pp,ns=timed_predictions(model,queries)
        counters=defaultdict(int);times=[];scan_times=[];typed_times=[]
        for qi,(row,(prob,hyp)) in enumerate(zip(queries,pp)):
            decision=max(range(7),key=lambda z:prob[z]);rank=sorted(range(6),key=lambda z:-prob[z])
            gold=row['gold'];entity=row['entity'];top=rank[0]
            start=perf_counter_ns(); ids,visits=idx.retrieve(entity,rank,k=1)
            best=store[ids[0]] if ids else None
            end=perf_counter_ns()
            times.append(end-start);counters['candidate_checks']+=visits
            counters['action_correct']+=int(decision==gold)
            counters['action_defer']+=int(decision==6)
            if gold!=6:
                counters['unambiguous']+=1
                counters['read_correct']+=int(best is not None and best['action']==gold and decision==gold)
                counters['top1_recall']+=int(top==gold)
                for k in BUDGETS:counters[f'top{k}_recall']+=int(gold in rank[:k])
                # Frozen target-precision gate: no probabilistic model output
                # carries write authority. This is an answer-only correctness check.
            else:
                counters['ambiguous']+=1
                counters['safe_defer']+=int(decision==6)
                if hyp is not None:
                    two=sorted(range(6),key=lambda z:-hyp[z])[:2]
                    counters['both_alternatives']+=int(set(two)==set(row['pair']))
            s=perf_counter_ns();item,inspected=idx.scan(entity,top);scan_times.append(perf_counter_ns()-s)
            assert item is not None and inspected>0
            s=perf_counter_ns();typed_result=idx.typed_read(entity,top);typed_times.append(perf_counter_ns()-s)
            assert typed_result==item
            if label=='multi':
                per_case.append(dict(seed=seed,n=n,request_hash=row['text_sha256'],gold=gold,
                                     predicted=decision,correct=bool(decision==gold),
                                     top1=top,alternatives=(sorted(range(6),key=lambda z:-hyp[z])[:2] if hyp is not None else [])))
        outputs[label]=dict(n=len(queries),**counters,read_us_median=statistics.median(times)/1000,
                            encoding_and_forward_us_median=statistics.median(ns)/1000,
                            pipeline_us_median=(statistics.median(times)+statistics.median(ns))/1000,
                            scan_us_median=statistics.median(scan_times)/1000,
                            typed_us_median=statistics.median(typed_times)/1000)
    # Safety and raw-history tests use no hidden task labels or action execution.
    sample_entity=names[0]
    assert idx.retrieve('missing-entity',[0],1)[0]==()
    assert len(idx.retrieve(sample_entity,[0],1)[0])==1
    idx.invalidate(sample_entity)
    assert idx.retrieve(sample_entity,[0],1)[0]==()
    assert len(idx.index[sample_entity])==6  # raw episodes remain auditable
    fresh=dict(entity=sample_entity,action=0,payload='new-signature',
               raw=f'archive {sample_entity} stage schedule entry new-signature',
               source_epoch=2,provenance='signed-simulated')
    append_ns=idx.append(fresh)
    ids,_=idx.retrieve(sample_entity,[0],1,epoch=2)
    assert len(ids)==1 and idx.rows[ids[0]]['payload']=='new-signature'
    assert len(idx.index[sample_entity])==7
    return dict(seed=seed,history=n,entity_count=len(names),build_ms=build_ns/1e6,
                postings=sum(len(v) for v in idx.postings.values()),raw_bytes=sum(len(r['raw']) for r in store),
                query_count=len(queries),controls=dict(controls),incremental_append_us=append_ns/1000,
                models=outputs,examples=per_case,
                limitations=['Queries provide an exact external entity key','Episodic observations contain one per-action entry: the action inventory is known',
                             'Held utterances are synthetic and drawn from earlier 091-100 grammar',
                             'The 6-action typed index can be faster and equally accurate for this known schema',
                             'Scans are timed across all 60 queries, but not matched in predictive quality to a perfect-oracle reader',
                             'Candidate top-k coverage is not automatically a correct answer',
                             'No actual COG authority token or signed commit is exercised'])


def run(seeds=SEEDS,sizes=SIZES):
    torch.set_num_threads(1)
    outcomes=[];learners=[]
    for seed in seeds:
        models={};reported={}
        for name,mode in [('parallel','parallel_specialists'),('transformer','self_attention')]:
            models[name],reported[name]=train_router(seed,mode)
        models['multi'],reported['multi']=train_multi(seed)
        learners.append(dict(seed=seed,models=reported))
        for n in sizes:
            result=evaluate(seed,n,models);outcomes.append(result)
            print('seed',seed,'history',n,'correct', {k:(v['action_correct'],v['read_correct'],v['safe_defer']) for k,v in result['models'].items()},flush=True)
    return dict(name='COG-ATTN-097a episodic + learned perceptual routing synthetic bridge',
                qualification='subsystem diagnostic only, no live COG ledger integration',
                seeds=list(seeds),sizes=list(sizes),training=learners,results=outcomes)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='097-results.json');p.add_argument('--seeds',default=','.join(map(str,SEEDS)))
    p.add_argument('--sizes',default=','.join(map(str,SIZES)))
    args=p.parse_args();x=run(tuple(map(int,args.seeds.split(','))),tuple(map(int,args.sizes.split(','))))
    with open(args.out,'w') as f:json.dump(x,f,indent=2,sort_keys=True)
