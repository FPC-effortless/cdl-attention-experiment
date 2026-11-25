"""No-fit initialized C learned-path router survival on the reviewed TRAIN only."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
from .data import resolve_prompt, sentences, WRITE_RE
from .fit import digest, read_train, read_vocab, stable_tensor_hash
from .model import build_decoder

SEEDS=(171,271,371)
WIDTH,LAYERS,BLOCK,RETAINED=32,2,8,64
BATCH_SIZE=8
AGE_CUTS=(64,128,256,512)


def locate(prompt,tok):
    """External text-to-offset alignment; the Decoder receives only tok.encode(prompt)."""
    value,n=resolve_prompt(prompt)
    pieces=sentences(prompt)
    source=pieces[n]
    match=WRITE_RE.fullmatch(source)
    if match is None or match[2]!=value:
        raise ValueError("cannot reconstruct the accepted source")
    start=len(tok.encode(" ".join(pieces[:n])))-1
    before=len(tok.encode(source[:match.start(2)]))-2
    stop=start+len(tok.encode(source))-2
    anchor=start+before
    ids=tok.encode(prompt)
    if ids[start:stop]!=tok.encode(source)[1:-1] or ids[anchor]!=tok.index[value] or ids[-1]!=5:
        raise ValueError("source span/color token or answer delimiter shifted")
    if not (0<start<=anchor<stop<len(ids)-1):
        raise ValueError("source token outside prior evidence")
    return ids,anchor,start,stop,value


def simulate(writers,publish,q,block,retained,topics=4,capacity=2):
    """Independent Python FIFO replay from the exact hidden-input writer decisions."""
    if len(writers)!=len(publish) or not 0<=q<len(writers) or topics<2 or capacity<1:
        raise ValueError("invalid native trace shape")
    book=[[] for _ in range(topics)]
    def expire(t):
        floor=t//block-retained+1
        for j,b in enumerate(book):
            book[j]=[p for p in b if p//block>=floor]
    def snapshot():
        return [b+[-1]*(capacity-len(b)) for b in book]
    for i,(w,on) in enumerate(zip(writers,publish,strict=True)):
        expire(i)
        if i==q:
            pre=snapshot()
        if on:
            book[w]=[i,*book[w]][:capacity]
    expire(len(writers))
    return pre,snapshot()


def read_two(row,topics):
    if len(topics)!=2 or len(set(topics))!=2:
        raise ValueError("subscriber must select two distinct topics")
    return [p for topic in topics for p in row[topic]]


def audit_trace(layer,x,in_state,ret,query):
    if in_state is not None or layer.training or layer.route_mode!="learned":
        raise ValueError("diagnostic requires fresh native learned-mode eval streams")
    out,post,native=layer.forward_with_candidates(x,in_state)
    native_out,native_post=ret
    for a,b in ((out,native_out),(post.values,native_post.values),
                (post.positions,native_post.positions),(post.cursor,native_post.cursor)):
        if not torch.equal(a,b):
            raise RuntimeError("route trace replay altered outputs or state")
    # Both project on identical complete input geometry: no prefix GEMM shape change.
    sender=layer._linear(layer.publisher,x).argmax(-1).tolist()
    admitted=(layer._linear(layer.admit,x).squeeze(-1)>=0).tolist()
    batch=torch.arange(x.shape[0],device=x.device)
    reader=layer._linear(layer.subscriber,x)[batch,query].tolist()
    if not torch.isfinite(x).all():
        raise ValueError("nonfinite router hidden input")
    learned_native=native[batch,query].reshape(x.shape[0],4).tolist()
    result=[]
    for i,(w,p,q) in enumerate(zip(sender,admitted,query.tolist(),strict=True)):
        pre,expected=simulate(w,p,q,layer.block,layer.retained_blocks,layer.TOPICS,layer.PER_TOPIC)
        # Independent Python stable top-two on the actual logits.
        selected=sorted(range(layer.TOPICS),key=lambda j:(-reader[i][j],j))[:2]
        selected_blind=(q%layer.TOPICS,(q+1)%layer.TOPICS)
        learned=read_two(pre,selected)
        if learned!=learned_native[i] or expected!=post.positions[i].tolist() or post.cursor[i]!=x.shape[1]:
            raise RuntimeError("independent FIFO/selection replay disagrees with native indices")
        result.append((p,pre,learned,read_two(pre,selected_blind),selected,selected_blind))
    return result


def project_counts(q,anchor,start,stop,route):
    p,retained,learned,blind,_,_=route
    age=lambda i:q//BLOCK-i//BLOCK<RETAINED
    cache={v for t in retained for v in t if v>=0}
    support=set(range(start,stop))
    def view(indices):
        em=sorted(i for i in indices if p[i])
        keep=sorted(indices&cache)
        sl=sorted(indices&set(learned));sb=sorted(indices&set(blind))
        if not (set(sl)<=set(keep) and set(sb)<=set(keep) and set(keep)<=set(em)):
            raise RuntimeError("survival accounting violated")
        return {'age_eligible':sum(age(i) for i in indices),'positions':len(indices),
                'emitted':em,'retained':keep,'learned_selected':sl,'blind_selected':sb}
    return {'anchor':view({anchor}),'sentence_any':view(support)}


def summarize(rows):
    """Unconditional per-world/per-seed counts plus paired-family denominators."""
    c=defaultdict(Counter);f=defaultdict(dict)
    for x in rows:
        if not 1<=x["anchor_age"]<=AGE_CUTS[-1]:
            raise ValueError("anchor age outside the admitted prompt geometry")
        bin_name=next(f"<={n}" for n in AGE_CUTS if x["anchor_age"]<=n)
        for view,data in x['views'].items():
            sample={k:bool(data[k]) for k in ('age_eligible','emitted','retained','learned_selected','blind_selected')}
            for age in ('all',bin_name):
                key=f'seed={x["seed"]}/layer={x["layer"]}/view={view}/anchor_age={age}'
                c[key]['worlds']+=1;c[key]['eligible_source_positions']+=data['age_eligible']
                c[key]['total_source_positions']+=data['positions']
                for k,v in sample.items():c[key][k+'_worlds']+=int(v)
                for k in ('emitted','retained','learned_selected','blind_selected'):
                    c[key][k+'_source_positions']+=len(data[k])
                p=f[key].setdefault(x['family_ordinal'],{})
                if x['world'] in p:
                    raise ValueError("duplicate per-family observation")
                p[x['world']]=sample
    for key,items in f.items():
        for pair in items.values():
            if set(pair)!={0,1}:
                raise ValueError("missing same-age family pair; no denominator substitution")
            c[key]['families']+=1
            for k in pair[0]:c[key][k+'_both_world_families']+=int(pair[0][k] and pair[1][k])
    return {key:dict(fields) for key,fields in sorted(c.items())}


@torch.inference_mode()
def collect(encoded,model,seed,batch_size=BATCH_SIZE,observations=None):
    """Only token tensor crosses native model interface; views joined after its call."""
    traces={}
    handles=[]
    for idx,layer in enumerate(model.mixers):
        if layer.arm!="C" or not layer.route_enabled:
            raise ValueError("must instrument exact C learned path")
        def hook(r,args,kw,ret,*,i=idx):
            if set(kw)!={'disable'} or kw['disable'] or i in traces or len(args)!=2:
                raise ValueError("unexpected route call contract")
            traces[i]=audit_trace(r,*args,ret,query)
            if observations is not None:
                observations["route_probe_forwards"]+=1
        handles.append(layer.route.register_forward_hook(hook,with_kwargs=True))
    try:
        ordered=sorted(encoded,key=lambda e:(len(e[2]),e[0],e[1]))
        rows=[]
        for i in range(0,len(ordered),batch_size):
            batch=ordered[i:i+batch_size]
            n=max(len(e[2]) for e in batch)
            query=torch.tensor([len(e[2])-1 for e in batch],dtype=torch.long)
            xx=torch.tensor([e[2]+[0]*(n-len(e[2])) for e in batch],dtype=torch.long)
            traces.clear()
            _=model(xx)
            if observations is not None:
                observations["model_forwards"]+=1
            if set(traces)!=set(range(len(model.mixers))):
                raise RuntimeError("one or more native layers were not observed")
            for k,(family,world,_,anchor,start,stop) in enumerate(batch):
                q=int(query[k])
                for level in range(len(model.mixers)):
                    evidence=traces[level][k]
                    rows.append({'family_ordinal':family,'world':world,'seed':seed,'layer':level,
                                 'delimiter_position':q,'anchor_position':anchor,
                                 'anchor_age':q-anchor,'source_span':[start,stop],
                                 'prequery_topic_positions':evidence[1],
                                 'learned_candidate_positions':evidence[2],
                                 'blind_candidate_positions':evidence[3],
                                 'learned_subscribed_topics':evidence[4],
                                 'blind_subscribed_topics':list(evidence[5]),
                                 'prompt_positions':q+1,'published_prompt_positions':sum(evidence[0][:q+1]),
                                 'views':project_counts(q,anchor,start,stop,evidence)})
        return sorted(rows,key=lambda r:(r['family_ordinal'],r['world'],r['layer']))
    finally:
        for h in handles:h.remove()


def run(train,vocab,train_sha256,vocab_sha256):
    t0=time.perf_counter()
    if digest(train)!=train_sha256 or digest(vocab)!=vocab_sha256:
        raise ValueError("TRAIN/derived vocabulary input digest mismatch")
    fam=read_train(train);tok=read_vocab(vocab,fam)
    if len(fam)!=320:
        raise ValueError("exact 320 reviewed TRAIN families required")
    encoded=[]
    for i,(prompts,answers) in enumerate(fam):
        duo=[]
        for j,p in enumerate(prompts):
            ids,anchor,start,stop,v=locate(p,tok)
            if v!=answers[j] or len(ids)>512:
                raise ValueError("sealed TRAIN annotation conflicts or input exceeds geometry")
            duo.append((len(ids),anchor,start,stop))
            encoded.append((i,j,ids,anchor,start,stop))
        if duo[0]!=duo[1]:
            raise ValueError("pair offset alignment failed")
    out=[];weights={};observed=Counter()
    for seed in SEEDS:
        model=build_decoder(tok.size,"C",width=WIDTH,layers=LAYERS,block=BLOCK,seed=seed,
                            route_mode="learned",route_enabled=True,retained_blocks=RETAINED).eval()
        initial=stable_tensor_hash(model.state_dict(),lambda _: True)
        out+=collect(encoded,model,seed,observations=observed)
        if stable_tensor_hash(model.state_dict(),lambda _: True)!=initial:
            raise RuntimeError("diagnostic modified initialized parameters")
        weights[str(seed)]=initial
    if (len(out)!=640*len(SEEDS)*LAYERS or observed["model_forwards"]!=240
        or observed["route_probe_forwards"]!=480):
        raise RuntimeError("incomplete seed/layer/world accounting")
    result={'schema':'cog-comm-001-initialized-survival.v1','claim_eligibility':'NOT_QUALIFIED',
            'source_scope':'TRAIN only; no fitter, CAL, DEV or protected TEST','fit_updates':0,'protected_inputs':0,
            'train_sha256':train_sha256,'vocab_sha256':vocab_sha256,'seed_initialization_sha256':weights,
            'families':len(fam),'worlds':len(encoded),'seeds':list(SEEDS),'layer_count':LAYERS,
            'world_seed_layer_trials':len(out),'per_layer_same_state_blind_scope':True,
            'right_padding':'deterministic length-sorted batches of 8; per-row true delimiter frontier before all padding','token_age_bins':list(AGE_CUTS),'width':WIDTH,'block':BLOCK,'retained_blocks':RETAINED,
            'model_forwards':observed['model_forwards'],'route_probe_forwards':observed['route_probe_forwards'],'layer_stats':summarize(out),'wall_seconds':time.perf_counter()-t0}
    return result,out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("train","vocab","output","trace"):
        p.add_argument('--'+n,type=Path,required=True)
    for n in ("train-sha256","vocab-sha256"):
        p.add_argument('--'+n,required=True)
    a=p.parse_args()
    if a.output.exists() or a.trace.exists() or a.output.resolve()==a.trace.resolve() or sys.flags.optimize:
        raise SystemExit("fresh distinct outputs and assert-enabled runtime required")
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    result,rows=run(a.train,a.vocab,a.train_sha256,a.vocab_sha256)
    with a.trace.open('x',encoding='utf8') as f:
        for r in rows:f.write(json.dumps(r,sort_keys=True,allow_nan=False,separators=(',',':'))+'\n')
    with a.trace.open('rb') as f:
        result['private_trace_sha256']=hashlib.file_digest(f,'sha256').hexdigest()
    with a.output.open('x',encoding='utf8') as f:
        json.dump(result,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')

if __name__=="__main__":main()
