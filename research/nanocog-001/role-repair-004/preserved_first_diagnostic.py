"""Retrospective selection from sealed outputs, not a learned policy or gate."""
import json
from pathlib import Path
from hashlib import sha256
A='6915a754aa598d5608f0041850b3897ecfeef0bd8731835815e24333ba6c98d1'
S='0971d030ad7b74f7456e5eabdfa39b87dcd3641d90b6c2dfbf9e31e51366f1da'
def read(p,d):
    b=Path(p).read_bytes()
    if sha256(b).hexdigest()!=d:raise ValueError('input changed')
    return json.loads(b)
s=read('REPAIR_PREDICTION_SEAL.json',S)['results'];a=read('repair_panel/actor.json',A)['prompts']
first=[q.split()[0]==q.split()[-1] for q in a]
def ok(k):
    p=Path('repair_actor_runs')/s[k]['id']/'output/predictions.json'
    z=read(p,s[k]['predictions_sha256'])
    assert z['actor_sha256']==A and len(z['logits'])==len(a)==960
    idx={k:i+262 for i,k in enumerate(z['tokenizer_words'])}
    out=[]
    for line,l in zip(a,z['logits'],strict=True):
        name,op,v,other,op2,w,q=line.split()
        assert op==op2=='is' and q in (name,other)
        j=idx[v if name==q else w]
        out.append(l[j]>max(v for i,v in enumerate(l) if i!=j))
    return out
n=ok('native-1371-0')
out={}
for arm in ('tap_compare','literal_control'):
    for seed in (1471,1571,1671):
        v=ok(f'{arm}-{seed}-1280')
        g=[x if f else y for x,y,f in zip(n,v,first,strict=True)]
        out[f'{arm}-{seed}']=dict(strict_worlds=sum(g),strict_families=sum(all(g[j:j+8]) for j in range(0,960,8)),
            strict_histories=sum(g[j] and g[j+1] for j in range(0,960,2)))
print(json.dumps(dict(kind='RETROSPECTIVE_ROLE_ORACLE_NOT_A_LEARNED_POLICY',scores=out),indent=2,sort_keys=True))
