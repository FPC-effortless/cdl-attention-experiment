"""Seal verified actor artifacts, then score with independent raw text answers."""
import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
import itertools
from trainlab.runner import verify_run

A='201a63921dfaaf601b03e03e7dc80858e9958b71fe0eef8e6b8572d8d459aac1'
J='ce80c30bfbb9c38ae8b078d16c0fd8187bc8a641bba4315bd05dad5253927118'
def h(p):return sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_bytes())
def require(t,msg):
    if not t:raise ValueError(msg)
def write(p,s):
    with p.open('x') as f:json.dump(s,f,allow_nan=False,indent=2,sort_keys=True)
def seal(ledger,pin,out):
    require(h(ledger)==pin,'actor ledger changed')
    rows=read(ledger); records={}
    require(len(rows)==12,'registered actor count')
    for cid,case in rows.items():
        d=ledger.parent/cid
        r=read(d/'cli-output.json')
        path=Path('actors')/r['id']
        require(verify_run(path)['valid'] and r==read(path/'run.json') and
                r['status']=='succeeded' and r['inputs']==case['inputs'] and
                r['name']==cid and r['seed']==case['seed'] and
                h(d/'recipe.json')==case['recipe_sha256'] and read(d/'recipe.json')==read(path/'experiment.json') and
                r['metrics']['prediction_rows']==960,'native identity')
        records[cid]=dict(id=r['id'],receipt_sha256=h(path/'run.json'),
                         prediction_sha256=r['artifacts']['predictions.json'])
    require(len({v['id'] for v in records.values()})==12,'duplicate actor')
    write(out,records)
    return h(out)
def judge(ledger,pin,seal,seal_pin,out):
    require(h(ledger)==pin and h(seal)==seal_pin,'ledger or seal changed')
    cases,results=read(ledger),read(seal)
    require(set(cases)==set(results) and len(cases)==12,'coverage')
    a=Path('panel/confirm_actor.json');j=Path('panel/confirm_judge.json')
    require(h(a)==A and h(j)==J,'wrong confirmation panel')
    prompts=read(a)['prompts']; fs=read(j)['families']
    recon=[]
    for i,f in enumerate(fs):
        idx=list(range(8*i,8*i+8));require(idx==f['rows'],'panel row alignment')
        z=[]
        for r in idx:
            name,v,x,other,w,y,q=prompts[r].split()
            require(v==w=='is' and name!=other and x!=y and q in (name,other),'malformed grammar')
            ans={name:x,other:y}[q];foil=y if ans==x else x
            require(f['answers'][r%8]==ans,'truth disagreement')
            z.append((ans,foil))
        recon.append((z,'seen_name_pair' if f['familiar_name_pair'] else 'unseen_name_pair'))
    slots=dict()
    for cid,s in results.items():
        c=cases[cid]; path=Path('actors')/s['id']
        require(h(path/'run.json')==s['receipt_sha256'],'receipt changed')
        path=path/'output/predictions.json'; raw=path.read_bytes()
        require(sha256(raw).hexdigest()==s['prediction_sha256'],'logits changed')
        p=json.loads(raw)
        require((p['actor_sha256'],p['checkpoint_sha256'],p['seed'],p['updates'],p['arm'])==
                (A,c['checkpoint_sha256'],c['seed'],c['endpoint'],'gated32'),'checkpoint association')
        vocab=read(ledger.parent/cid/'vocabulary.json')
        require(h(ledger.parent/cid/'vocabulary.json')==c['inputs']['vocabulary.json'] and
                p['tokenizer_words']==vocab['words'] and p['vocab_size']==vocab['size']==287 and
                len(p['logits'])==960,'vocabulary or output coverage')
        ids={v:i+262 for i,v in enumerate(vocab['words'])}
        total={k:Counter() for k in ('all','seen_name_pair','unseen_name_pair')}
        for i,(query,group) in enumerate(recon):
            exact,rel,margin=[],[],[]
            for row,(answer,foil) in zip(p['logits'][i*8:i*8+8],query,strict=True):
                require(len(row)==287 and all(type(x) in [int,float] and math.isfinite(x) for x in row),'bad logits')
                y=ids[answer]; correct=row[y]
                exact.append(all(correct>v for k,v in enumerate(row) if k!=y))
                rel.append(correct>row[ids[foil]])
                margin.append(correct-row[ids[foil]])
            row=dict(worlds=8,families=1,exact_worlds=sum(exact),exact_families=all(exact),
                     two_color_worlds=sum(rel),two_color_families=all(rel),
                     two_color_ties=sum(x==0 for x in margin))
            for g in ('all',group):total[g].update(row)
        slots[cid]=total
    write(out,slots)
if __name__=='__main__':
    import sys
    mode,ld,pin,ar,*extra=sys.argv[1:]
    if mode=='seal':
        print(seal(Path(ld),pin,Path(ar)))
    elif mode=='judge':
        judge(Path(ld),pin,Path(ar),extra[0],Path(extra[1]))
    else:raise ValueError('unknown stage')
