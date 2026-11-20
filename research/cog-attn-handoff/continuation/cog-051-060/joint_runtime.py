"""Compare five deployed controllers using the same authenticated typed route and physical planner."""
from collections import Counter
import json
from pathlib import Path
import sys
import joblib
import numpy as np
import torch
import onnxruntime as ort
from transformers import AutoTokenizer
from scipy.special import softmax
from grounding import NAMES,TRAIN,HELD,Sequence,role_features
from panel import Panel,probe,plan
from system_tests import bridge

HERE=Path(__file__).resolve().parent
D=Path('/workspace/cog-next-private')
PRIOR=Path('/workspace/delta-authority-048')
sys.path.insert(0,str(PRIOR/'recipe'))
import bridge_scope as original
from scope import Scope

def entail(texts,hyps):
    tokenizer=AutoTokenizer.from_pretrained('/workspace/issue-history-study/recipe/nli',local_files_only=True)
    cfg=ort.SessionOptions();cfg.intra_op_num_threads=1;cfg.inter_op_num_threads=1
    session=ort.InferenceSession('/workspace/issue-history-study/recipe/nli/model_quint8_avx2.onnx',cfg,providers=['CPUExecutionProvider'])
    ids={x.name for x in session.get_inputs()};scores=[]
    for i in range(0,len(texts),32):
        encoded=tokenizer(texts[i:i+32],hyps[i:i+32],padding=True,return_tensors='np')
        scores.extend(softmax(session.run(None,{k:v for k,v in encoded.items() if k in ids})[0],axis=1)[:,1])
    return np.array(scores).reshape(-1,2)

def run(seed):
    root=D/'lab/runs'
    folders=[p.parent for p in root.glob('*/run.json') if (lambda d:d['name']=='acquisition-candidate' and d['seed']==seed)(json.loads(p.read_text()))]
    src=sorted(folders)[0]/'output'
    bindings=joblib.load(src/'roles.joblib')
    vocab=bindings['vocab'];rev=bindings['dictionary'];word={a:w for w,a in rev.items() if w in NAMES}
    rng=np.random.default_rng(seed+290000)
    tasks=[];xs=[];features=[];hyps=[];prem=[]
    for i in range(512):
        a,b=map(int,rng.choice(5,2,replace=False))
        phrase=rng.choice(TRAIN) if i%3==0 else HELD[i%3-1]
        raw=phrase.format(p=word[a],q=word[b])
        canonical=[NAMES[rev[t]] if t in rev else t for t in raw.split()]
        names=[t for t in canonical if t in NAMES]
        features.extend(role_features(canonical,t) for t in names)
        prem.extend([raw,raw])
        hyps.extend(f'The requested change applies to {word[NAMES.index(t)]}.' for t in names)
        tasks.append((a,b,raw,names))
        token=[vocab.get(t,1) for t in canonical];xs.append(token+[0]*(16-len(token)))
    x=torch.tensor(xs)
    score=bindings['clf'].decision_function(bindings['vec'].transform(features)).reshape(-1,2)
    evidence=entail(prem,hyps)
    candidate_ids=np.array([[NAMES.index(y) for y in r[-1]] for r in tasks])
    def pick(scores):return candidate_ids[np.arange(len(tasks)),scores.argmax(1)]
    preds={'local_roles':pick(score),'assisted_roles':pick(score+np.log(evidence.clip(1e-6))),
           'pretrained_only':pick(evidence)}
    nets={}
    for kind in ('gru','transformer'):
        net=Sequence(max(vocab.values())+1,kind)
        net.load_state_dict(torch.load(src/f'{kind}.pt',weights_only=True));nets[kind]=net.eval()
        with torch.no_grad(): preds[kind]=net(x).argmax(1).numpy()
    csm=original.reader.Reader(seed,True)
    w=np.load(src/'csm.npz')
    csm.E,csm.a,csm.b=w['E'],w['a'],w['b']
    refs=np.zeros((5,16),int)
    for i,n in enumerate(NAMES):refs[i,0]=vocab[n]
    preds['csm']=np.argmax(csm.forward(x.numpy())[0]@csm.forward(refs)[0].T,axis=1)
    routes=bridge(seed+0,original,Scope,PRIOR)['results']['integrated']
    # Induce a reusable utterance from one verified example. Rebind its grounded words for a new requested referent.
    utterance=next(iter(bindings['utterances'].values()))
    text,src_id=utterance
    utter=[];target=[]
    for a,b,raw,names in tasks:
        parts=[NAMES[a if rev[t]==src_id else b] if t in rev else t for t in text.split()]
        token=[vocab.get(t,1) for t in parts];utter.append(token+[0]*(16-len(token)));target.append(a)
    with torch.no_grad():listener=nets['transformer'](torch.tensor(utter)).argmax(1).numpy()
    summary={}
    for arm,actions in preds.items():
        local=np.random.default_rng(seed+270000);env=Panel(local);model=probe(env,local,True)
        state=0;cache={};world=original.World()
        n_correct=mismatch=hits=db_errors=steps=0
        for i,((goal,_,_,_),action) in enumerate(zip(tasks,actions)):
            if i in (160,320):
                env=Panel(local)
                if i==160: model=probe(env,local,True);cache.clear();world.advance()
            initial=state;encoded=model.state(env.frame(state));key=encoded,encoded^(1<<int(action))
            route=routes[i%len(routes)]
            hit=key in cache;hits+=hit
            path=cache[key] if hit else plan(model,*key)
            good=path is not None and route['admitted']
            if good:
                for button in path:
                    expected=model.graph[model.state(env.frame(state))][button]
                    state=env.step(state,button);steps+=1
                    if model.state(env.frame(state))!=expected:
                        good=False;mismatch+=1;model=probe(env,local,True);cache.clear();world.advance();break
            if good:
                cache[key]=path
                status=world.commit(str(i),world.epoch(),{str(route['place']):state})
                db_errors+=status!='committed'
            n_correct+=bool(good and route['exact'] and state==(initial^(1<<goal)))
        world.db.close()
        summary[arm]=dict(turns=len(tasks),correct=n_correct,repaired=mismatch,cache_hits=hits,steps=steps,db_errors=db_errors)
    return dict(communication_correct=int((listener==target).sum()),communication_n=len(tasks),arms=summary)
if __name__=='__main__':
    torch.set_num_threads(1)
    (HERE/'results/joint-runtime.json').write_text(json.dumps({s:run(s) for s in (5101,5107,5113)},default=lambda x:x.item()))
