"""Seal exact frozen feature arrays, then fit only independently specified probes."""
from hashlib import sha256
from pathlib import Path
from collections import defaultdict
import math,json,random,sys,os
import numpy as np
from trainlab.runner import verify_run
from probe_core import fit,metrics
LEDGER='b78161b7687aa0d89c86e5192dd2863ca7542b5e905d2ccb4836143421fd4425'
A='653460a4ead17c87e5b17de9a10b7bb17cdfdedff131a3afca2a182528030b35'
J='0ce6f4cb86daa3edd9f09271c564fe9f28320e25f78941db56f9356f62b28391'
NAMES=tuple("ada ben cy di eli fay gio hal ivy jay kai lou mia nik ola pam".split())
COLORS=tuple("blue gold green orange pink purple red white".split())
STAGES=(('literal_raw',32),('literal_normalized',32),('l2_gate_input',32),('l2_gate_probability',4),('pre_output_hidden',32))
P=Path('trace_admissions')
R=Path('trace_runs')
def need(ok,msg):
    if not ok:raise ValueError(msg)
def b(p,pin=None):
    s=p.read_bytes();need(pin is None or sha256(s).hexdigest()==pin,'file pin '+str(p));return s
def h(p):return sha256(b(p)).hexdigest()
def read(p,pin=None):return json.loads(b(p,pin))
def save(p,x):
    with p.open('x') as f:json.dump(x,f,sort_keys=True,allow_nan=False,indent=2);f.write('\n')
    print(h(p))
def seal():
    led=read(P/'ledger.json',LEDGER);out={}
    need(set(led)=={'trace-1171','trace-1371'},'actor roster')
    for name,c in led.items():
        case=P/name;r=read(case/'cli-output.json');run=R/r['id']
        need(verify_run(run)['valid'] and r==read(run/'run.json') and r['status']=='succeeded' and
             r['name']==name and r['seed']==c['seed'] and r['inputs']==c['inputs'] and
             r['metrics']==dict(trace_rows=1536,model_forwards=192,optimizer_updates=0,response_logit_rows=0) and
             set(r['artifacts'])=={'TRACE.f32le','trace_contract.json'},'native scope or receipt')
        need(h(case/'recipe.json')==c['recipe_sha256'] and all(h(case/k)==v for k,v in c['inputs'].items()),'actor pin')
        out[name]=dict(id=r['id'],receipt_sha256=h(run/'run.json'),artifacts=r['artifacts'])
    save(Path('TRACE_SEAL.json'),dict(ledger_sha256=LEDGER,runs=out))

def panel():
    prompts=read(Path('trace_panel/actor.json'),A)['prompts']
    f=read(Path('trace_panel/judging_contract.json'),J)['families']
    need(len(f)==192 and len(prompts)==len(set(prompts))==1536,'panel')
    targets={t:np.empty(1536,dtype=np.int64) for t in ('role','value','identity','foil')}
    ids={'probe_fit':[],'probe_held':[]};keys={'probe_fit':set(),'probe_held':set()}
    families={k:[] for k in ids};held_group=defaultdict(list);used=set()
    for g in f:
        side=g['split'];k=tuple(g['name_pair'])
        need(len(g['cells'])==8,'factorial family')
        local=[]
        for a in g['cells']:
            i=a['row'];need(i not in used and 0<=i<1536,'row duplication')
            used.add(i); local.append(i)
            n,sep,v,m,sep2,w,q=prompts[i].split()
            need(sep==sep2=='is' and set((n,m))==set(k) and q in (n,m) and {v,w}==set(g['colors']),'grammar or lineage')
            t,foil=(v,w) if q==n else (w,v)
            need((t,foil,q,int(q==n))==(a['answer'],a['foil'],a['query_name'],a['query_in_rendered_first']),'truth')
            targets['role'][i]=q==n;targets['value'][i]=COLORS.index(t)
            targets['foil'][i]=COLORS.index(foil);targets['identity'][i]=NAMES.index(q)
        ids[side].extend(local);keys[side].add(k);families[side].append(local)
        if side=='probe_held':held_group[k].extend(local)
    need(len(used)==1536 and not keys['probe_fit']&keys['probe_held'] and
         (len(ids['probe_fit']),len(ids['probe_held']),len(held_group))==(1024,512,16),'group split')
    return targets,ids,families,held_group

def interval(group_values):
    r=random.Random(20261131);n=len(group_values)
    xs=sorted(sum(r.choices(group_values,k=n))/n for _ in range(2000))
    return [xs[50],xs[1949]]

def audit(pin):
    seal=read(Path('TRACE_SEAL.json'),pin); led=read(P/'ledger.json',LEDGER)
    need(seal['ledger_sha256']==LEDGER and set(seal['runs'])==set(led),'sealed roster')
    target,i,f,groups=panel()
    fitids=i['probe_fit'];ev=i['probe_held']
    rowmap={j:k for k,j in enumerate(ev)}
    held_pos={g:[rowmap[j] for j in v] for g,v in groups.items()}
    out={}
    for lab,x in seal['runs'].items():
        d=R/x['id']; r=read(d/'run.json',x['receipt_sha256']); a=x['artifacts']
        need(r['artifacts']==a,'output seal')
        s=read(d/'output/trace_contract.json',a['trace_contract.json'])
        need(s['shape']==[1536,132] and
             s['columns']==[{'name':n,'width':w} for n,w in STAGES] and s['trace_has_truth'] is False and
             s['checkpoint_sha256']==led[lab]['checkpoint_sha256'] and s['actor_sha256']==A
             and s['seed']==led[lab]['seed'] and s['trace_sha256']==a['TRACE.f32le'],'trace binding')
        mat=np.frombuffer(b(d/'output/TRACE.f32le',a['TRACE.f32le']),dtype='<f4').reshape(1536,132).astype(np.float64)
        q=mat[:,32:64]; columns=dict(prior=np.empty((1536,0)),q_only=q,literal_raw=mat[:,:32])
        cur=0
        for name,width in STAGES:
            vals=mat[:,cur:cur+width];cur+=width
            if not name.startswith('literal_'):
                columns[f'q_plus_{name}']=np.concatenate((q,vals),axis=1)
        group={}
        for t,cl in [('role',2),('value',8),('identity',16)]:
            y=target[t];fy=y[fitids];evy=y[ev];ch={}
            # Fixed within-family TRAIN label shuffle preserves support, never touches held.
            zz=y.copy();rr=random.Random(20261131)
            for fam in f['probe_fit']:
                prev=list(zz[fam]);rr.shuffle(prev);zz[fam]=prev
            base_n=None
            use=tuple(columns) if t!='identity' else ('prior','q_only','q_plus_l2_gate_input','q_plus_l2_gate_probability','q_plus_pre_output_hidden')
            for stage in use:
                feature=columns[stage]
                prob,run=fit(feature[fitids],fy,feature[ev],cl,prior=(stage=='prior'))
                met,correct=metrics(prob,evy,target['foil'][ev] if t=='value' else None)
                ce=-prob[np.arange(512),evy]
                if stage=='q_only':base_n=ce
                delta=None if base_n is None else interval([float(np.mean(ce[v]-base_n[v])) for v in held_pos.values()])
                # Actual factorial row IDs are retained; two positions never go IID.
                flags={j:bool(w) for j,w in zip(ev,correct,strict=True)}
                pairs={k:0 for k in ('query','order','assignment')}
                for family in f['probe_held']:
                    for k,mask in [('query',1),('order',2),('assignment',4)]:
                        pairs[k]+=sum(flags[family[j]] and flags[family[j^mask]] for j in range(8) if not j&mask)
                null=None
                if stage!='prior':
                    nprob,nfit=fit(feature[fitids],zz[fitids],feature[ev],cl)
                    null=dict(metrics=metrics(nprob,evy,target['foil'][ev] if t=='value' else None)[0],fit=nfit)
                ch[stage]=dict(metrics=met,fit=run,paired_both_correct=pairs,
                              ce_delta_vs_q_only_group_95_nats=delta,label_shuffle=null)
            group[t]=ch
        out[lab]=group
    save(Path('INFO001_PROBE_RESULTS.json'),dict(trace_seal_sha256=pin,held_name_pair_groups=16,held_rows=512,
         probe_fit_rows=1024,stagewise=out,model_updates=0,native_logits_scored=False))
if __name__=='__main__':
    if sys.argv[1]=='seal':seal()
    elif sys.argv[1]=='score':audit(sys.argv[2])
    else:raise ValueError('bad operation')
