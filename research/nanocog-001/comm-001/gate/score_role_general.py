"""Weight-free replay scorer; only consume already pinned native predictions."""
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_bytes())
def h(p):return sha256(p.read_bytes()).hexdigest()
def req(x,msg):
    if not x:raise ValueError(msg)

def score(folder,seal,expected,ledger_path=Path('actor-admissions/ledger.json'),
          ledger_pin='9f53d947e7c81950e52368c2a6f71aee2362551c9d3676cfc011cf707413f970',
          fit_root=Path('fits-gated32')):
    req(h(seal)==expected,'seal changed')
    from trainlab.runner import verify_run
    audit_path=Path('role-data-audit/audit.json')
    req(h(ledger_path)==ledger_pin,'actor ledger changed')
    req(h(audit_path)=='674e5125633ed84a4d252763916f24893036e1a956fb85f4a3193731f00c583b','data audit changed')
    ledger=read(ledger_path)
    pins=read(audit_path)['input_hashes']
    out={}
    for cid,rec in read(seal).items():
        tr=ledger[cid]
        run=folder/rec['id']
        req(verify_run(run)['valid'],'actor native integrity')
        rr=read(run/'run.json')
        if 'run_sha256' in rec: req(h(run/'run.json')==rec['run_sha256'],'sealed actor receipt changed')
        req(rr['status']=='succeeded' and rr['seed']==tr['seed'] and rr['name']==cid
            and rr['inputs']==tr['inputs'] and rr['artifacts']['predictions.json']==rec['prediction_sha256'],'actor native binding')
        predecessor=fit_root/tr['source_run']
        req(verify_run(predecessor)['valid'],'fit native integrity')
        pr=read(predecessor/'run.json')
        req(pr['status']=='succeeded' and pr['research_adequacy']=='EXECUTED_CRITERIA_MET'
            and pr['seed']==tr['seed']
            and pr['artifacts'][f"checkpoint-{tr['step']}.pt"]==tr['checkpoint_sha256'],'fit native binding')
        pred=run/'output/predictions.json'
        payload=pred.read_bytes()
        req(sha256(payload).hexdigest()==rec['prediction_sha256'],'prediction bytes changed')
        p=json.loads(payload)
        req((p['actor_sha256'],p['checkpoint_sha256'],p['seed'],p['updates'],p['arm']) ==
            (tr['actor_sha256'],tr['checkpoint_sha256'],tr['seed'],tr['step'],tr.get('arm','gated32')),'run identity')
        split=tr['split']
        vocab=fit_root/tr['source_run']/'output/vocabulary.json'
        req(h(vocab)==pr['artifacts']['vocabulary.json'],'fit vocab integrity')
        vv=read(vocab)
        req(p['tokenizer_words']==vv['words'] and p['vocab_size']==vv['size']==287,'exact vocab')
        ix={w:i+262 for i,w in enumerate(vv['words'])}
        actor=Path('panels')/f'{split}_actor.json'
        judge=Path('panels')/f'{split}_judge.json'
        req(h(judge)==pins[judge.name] and h(actor)==pins[actor.name]==tr['actor_sha256'],'audited projections')
        worlds=read(actor)['prompts']
        target=read(judge)['families']
        req(len(worlds)==len(p['logits'])==len(target)*8,'coverage')
        exposure=fit_root/tr['source_run']/'output/schedule.json'
        req(h(exposure)==pr['artifacts']['schedule.json'],'schedule integrity')
        train_used=set(read(exposure)['family_ordinals'][:tr['step']])
        sums={'all':Counter()}
        for k,f in enumerate(target):
            rows=f['rows']
            req(rows==list(range(k*8,(k+1)*8)),'invalid family index')
            evals=[]
            signs=[]
            ties=0
            for j in rows:
                a,sep,b,c,sep2,d,q=worlds[j].split()
                req(sep==sep2=='is' and a!=c and b!=d and q in (a,c),'invalid source grammar')
                ans={a:b,c:d}[q]
                req(ans==f['answers'][j%8],'answer differs')
                logits=p['logits'][j]
                req(len(logits)==287 and all(type(v) in (float,int) and math.isfinite(v) for v in logits),'invalid logits')
                y=ix[ans]; foil=ix[d if ans==b else b]; v=logits[y]; can=f['key'][1]
                m=logits[ix[can[0]]]-logits[ix[can[1]]]
                signs.append((m>0)-(m<0)); ties+=m==0
                evals.append((all(v>t for i,t in enumerate(logits) if i!=y),v>logits[foil]))
            exact, relative=zip(*evals)
            defs={'correct_role':lambda s,o,q:s*q,'inverse_role':lambda s,o,q:-s*q,
                  'first_position':lambda s,o,q:s*o,'last_position':lambda s,o,q:-s*o,
                  'canonical_first':lambda s,o,q:s,'canonical_last':lambda s,o,q:-s,
                  'color_first':lambda s,o,q:1,'color_last':lambda s,o,q:-1}
            import itertools
            cells=list(itertools.product((-1,1),repeat=3))
            row=dict(families=1,worlds=8,exact_worlds=sum(exact),exact_families=all(exact),
                     relative_worlds=sum(relative),relative_families=all(relative),relative_ties=ties,
                     query_both_exact=sum(exact[i] and exact[i+1] for i in (0,2,4,6)),
                     order_both_exact=sum(exact[i] and exact[i+2] for i in (0,1,4,5)),
                     assignment_both_exact=sum(exact[i] and exact[i+4] for i in (0,1,2,3)))
            for z,g in defs.items():
                row['agreements_'+z]=sum(t==g(*c) for t,c in zip(signs,cells))
            label='optimizer_seen' if k in train_used else 'optimizer_unseen'
            if split=='control':label='seen_name_pair' if f['familiar_name_pair'] else 'unseen_name_pair'
            sums['all'].update(row);sums.setdefault(label,Counter()).update(row)
        out[cid]=sums
    req(set(out)==set(ledger),'did not score the exact complete registry')
    return out
if __name__=='__main__':
    raw=score(Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3])
    with Path(sys.argv[4]).open('x') as f:json.dump(raw,f,allow_nan=False,indent=2,sort_keys=True)
