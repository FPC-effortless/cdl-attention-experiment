import json
from pathlib import Path
from time import perf_counter_ns
import numpy as np
import torch
from torch.nn.functional import scaled_dot_product_attention
from panel import Panel, probe, plan


def bridge(seed,mod,Scope,root):
    from trainlab.runner import verify_run
    original={5101:1181,5107:1201,5113:1217}[seed]
    slot=mod.SEEDS.index(original)
    row=next(m for m in json.loads((root/'module-manifest.json').read_text()) if m['seed']==original and m['arm']=='candidate')
    run=root/'ecosystem/research_lab/runs'/row['run_id']
    assert verify_run(run)['valid']
    tac=mod.TAC().eval()
    tac.load_state_dict(torch.load(run/'output/tac.pt',weights_only=True))
    csm=mod.reader.Reader(original,True)
    w=np.load(run/'output/csm.npz');csm.E,csm.a,csm.b=w['E'],w['a'],w['b']
    w=np.load(root/f'portable/scope-{original}.npz')
    scope=Scope(list(w['names']),w['coef'],float(w['intercept']))
    result=mod.held(seed,tac,csm,'challenge',(11,23,37)[slot],901+slot,scope)
    return result


def persistent(seed,language,routes,World):
    rng=np.random.default_rng(seed+90000)
    world=World()
    env=Panel(rng)
    model=probe(env,rng,True)
    state=0
    cache={}
    records=[]
    for i,task in enumerate(language['joint']):
        if i in (50,100):
            env=Panel(rng)
            if i==50: # publicly announced switchboard revision
                model=probe(env,rng,True);cache.clear();world.advance()
        route=routes[i]
        perceived=model.state(env.frame(state))
        goal=perceived^(1<<task['action'])
        key=perceived,goal
        hit=key in cache
        path=cache[key] if hit else plan(model,perceived,goal)
        initial=state
        mismatch=False
        attempts=0
        if route['admitted'] and path is not None:
            for a in path:
                expected=model.graph[model.state(env.frame(state))][a]
                state=env.step(state,a); attempts+=1
                if expected!=model.state(env.frame(state)):
                    mismatch=True
                    break
            if mismatch:
                model=probe(env,rng,True); cache.clear();world.advance()
            else:
                cache[key]=path
                world.commit(str(i),world.epoch(),{str(route['place']):state})
        records.append(dict(turn=i,initial=initial,final=state,expected=initial^(1<<task['true']),
                            exact=bool(route['exact'] and not mismatch and state==initial^(1<<task['true'])),
                            route=route['exact'],path=path,attempts=attempts,hit=hit,repaired=mismatch))
    world.db.close()
    return records


def scaling(seed):
    # Every token contributes to a group statistic. A scoped control may maintain the same sufficient statistic.
    torch.set_num_threads(1)
    rng=np.random.default_rng(seed)
    results=[]
    for size in (128,2048,32768,131072,1048576):
        vals=rng.integers(0,2,size,dtype=np.uint8)
        t=perf_counter_ns()
        blocks=np.array_split(vals,32)
        total=[int(b.sum()) for b in blocks]
        count=[len(b) for b in blocks]
        build=perf_counter_ns()-t
        t=perf_counter_ns()
        tensors=[torch.tensor(b,dtype=torch.float32).reshape(1,1,-1,1) for b in blocks]
        zeros=[torch.zeros_like(z) for z in tensors]
        sums=[float(z.sum()) for z in tensors]
        att_build=perf_counter_ns()-t
        q=torch.zeros(1,1,1,1)
        rows=[]
        for i in range(256):
            k=int(rng.integers(32));j=int(rng.integers(count[k]))
            change=int(i%8==0)
            t=perf_counter_ns()
            if change:
                old=int(blocks[k][j]);blocks[k][j]=1-old;total[k]+=1-2*old
            val=total[k]/count[k]
            verified=0<=val<=1
            indexed=perf_counter_ns()-t
            t=perf_counter_ns()
            if change: tensors[k][0,0,j,0]=1-old
            av=float(scaled_dot_product_attention(q,zeros[k],tensors[k]))
            assert 0<=av<=1.00001
            attention=perf_counter_ns()-t
            t=perf_counter_ns()
            if change: sums[k]+=1-2*old
            cached=sums[k]/count[k]
            assert 0<=cached<=1
            cache_ns=perf_counter_ns()-t
            oracle=float(blocks[k].sum())/count[k]
            rows.append([indexed,attention,cache_ns,bool(verified and abs(val-oracle)<1e-9 and abs(av-oracle)<1e-5 and abs(cached-oracle)<1e-9)])
        results.append(dict(size=size,rows=rows,build_ns=build,attention_build_ns=att_build,
                            array_bytes=vals.nbytes,torch_value_and_key_bytes=8*size))
    return results
