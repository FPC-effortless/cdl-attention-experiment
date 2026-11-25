"""Seal native immutable INFO002 output; separately score, no models or trainable probes."""
from collections import Counter,defaultdict
from hashlib import sha256
from itertools import product
from math import fsum,isfinite
from pathlib import Path
from random import Random
import json,struct,sys
from trainlab.runner import verify_run

ROOT=Path(__file__).resolve().parent
LEDGER='425737b22fb1916dacedaba9aa630c779cb9be193e03844849ab352f4f208304'
PROTOCOL='aca2abd99722a0a69b954ceb8c12411df7f70e1c9a3d9b8734b186cfcce96e21'
A='41e1c19a805ac0061252113303d4bf4e5b0e2a3a3c582df4ba67f5763d858761'
J='3c403237f504fc4a40baa1fea144e00263a1d81ef67c03c6a99eee0e1c9a38f1'
P=ROOT/'info002_admissions'; R=ROOT/'info002_runs'
BR=('base','edit_first_native','edit_second_native','edit_first_gate','edit_first_taps','edit_first_joint',
    'edit_second_gate','edit_second_taps','edit_second_joint')
KINDS=('native','gate','taps','joint')
ROSTER={'info002-1171','info002-1371'}
METRICS=dict(model_forwards=416,optimizer_updates=0,baseline_rows=256,natural_rows=768,intervention_rows=1536,
             same_row_controls=768,earlier_hybrid_batches_exact=192)
ONE=struct.Struct('<287f')
def need(ok,msg):
    if not ok:raise ValueError(msg)
def sha(p):
    return sha256(p.read_bytes()).hexdigest()
def bytes_at(p,pin=None):
    b=p.read_bytes();need(pin is None or sha256(b).hexdigest()==pin,f'exact bytes: {p}');return b
def read(p,pin=None):return json.loads(bytes_at(p,pin))
def write(p,v):
    with p.open('x') as f:json.dump(v,f,sort_keys=True,allow_nan=False,indent=2);f.write('\n')
    return sha(p)
def mean(v):
    v=tuple(v);return fsum(v)/len(v)
def ci(vals):
    need(len(vals)==32,'bootstrap unit count')
    rng=Random(20261202)
    est=sorted(fsum(vals[int(rng.random()*32)] for _ in range(32))/32 for _ in range(2000))
    return [est[50],est[1949]]
def strict(v,ans):
    m=max(v)
    return v[ans]==m and v.count(m)==1
def check_story(co,story,input_row,artifacts):
    need(co['schema']=='cog-info-002.hybrid.v1' and co['seed']==input_row['seed'] and
         co['checkpoint_sha256']==input_row['checkpoint_sha256'] and co['actor_sha256']==A
         and co['shape']==[9,256,287] and tuple(co['branch_order'])==BR and co['logits_sha256']==artifacts['LOGITS.f32le'],
         'actor receipt bindings')
    need((co['earlier_hybrid_batches_exact'],co['same_row_batches_exact'],co['native_replay_batches_exact'],
          co['native_mix_recompute_batches_exact'])==(192,96,32,288) and
          len(set(co['tokenizer_words']))==25 and len(co['tokenizer_words'])==25,'runtime scope/parity')
    need(story['metrics']==METRICS,'native measured scope')
def seal():
    led=read(P/'ledger.json',LEDGER); need(set(led)==ROSTER,'reviewed actor roster')
    result={}
    for lab,info in led.items():
        box=P/lab;cli=read(box/'cli-output.json'); run=R/cli['id'];r=read(run/'run.json')
        need(verify_run(run)['valid'] and cli==r and r['name']==lab and r['seed']==info['seed'] and
             r['status']=='succeeded' and r['exit_code']==0 and 0<r['elapsed_seconds']<=45 and
             r['inputs']==info['inputs'] and set(r['artifacts'])=={'LOGITS.f32le','intervention_receipt.json'},'native scope')
        need(read(box/'recipe.json',info['recipe_sha256'])==read(run/'experiment.json')
             and info['protocol_sha256']==PROTOCOL==sha(box/'INFO002_PROTOCOL.md')
             and all(sha(box/k)==v for k,v in info['inputs'].items()) and 'judge.json' not in r['inputs'],'exact closure')
        ar=r['artifacts'];co=read(run/'output/intervention_receipt.json',ar['intervention_receipt.json'])
        check_story(co,r,info,ar)
        need((run/'output/LOGITS.f32le').stat().st_size==9*256*ONE.size,'logit shape bytes')
        result[lab]={'run_id':r['id'],'run_sha256':sha(run/'run.json'),'artifacts':ar}
    need({a['run_id'] for a in result.values()}=={p.name for p in R.iterdir() if p.is_dir()},'unexpected duplicate or extra run')
    print(write(ROOT/'INFO002_SEAL.json',dict(ledger_sha256=LEDGER,registered_protocol_sha256=PROTOCOL,runs=result)))

def panel():
    a=read(ROOT/'panel/actor.json',A);j=read(ROOT/'panel/judge.json',J)
    need(set(a)=={'schema','prompts','groups'} and a['schema']=='cog-info-002.actor.v1' and j['schema']=='cog-info-002.judge.v1'
         and len(a['prompts'])==len(set(a['prompts']))==768 and len(a['groups'])==len(j['groups'])==32,'panel scope')
    allrows=[];out=[];names_seen=set()
    for g,(ac,an) in enumerate(zip(a['groups'],j['groups'],strict=True)):
        keys=tuple(an['name_pair'])
        need(an['group']==g and keys not in names_seen and len(an['cells'])==8,'paired cluster')
        names_seen.add(keys); cache={}
        for i,c in enumerate(an['cells']):
            r={k:ac[k][i] for k in ('base','edit_first','edit_second')};allrows+=list(r.values())
            need(c['rows']==r,'row association')
            base,first,second=[a['prompts'][r[k]].split() for k in ('base','edit_first','edit_second')]
            need(len(base)==len(first)==len(second)==7 and
              [k for k in range(7) if base[k]!=first[k]]==[2] and [k for k in range(7) if base[k]!=second[k]]==[5],
              'matched edit degree')
            n,w,x,m,w2,y,q=base; need(w==w2=='is' and {n,m}==set(keys) and q in keys and x!=y,'grammar')
            assign=int(dict(((n,x),(m,y)))[keys[0]]!=an['baseline_colors'][0])
            order=int(n!=keys[0]); qix=int(q!=keys[0])
            tq='edit_first' if q==n else 'edit_second'; dq='edit_second' if q==n else 'edit_first'
            gold,other=(x,y) if q==n else (y,x);new=first[2];slot='first' if q==n else 'second'
            need(new==second[5] and new not in (x,y) and set((x,y,new))==set(an['color_triple']) and
              (assign,order,qix,slot,q,gold,other,new,tq,dq)==tuple(c[k] for k in
              ('assignment','order','query_index','query_slot','query_name','base_answer','distractor_value','inserted_color','target_variant','distractor_variant')),
              'independent target/distractor semantics')
            k=(assign,order,qix);need(k not in cache,'duplicate factorial')
            cache[k]=(g*8+i,slot,gold,other,new,tq,dq)
        need(set(cache)==set(product((0,1),repeat=3)),'full factorial')
        out.append(cache)
    need(sorted(allrows)==list(range(768)),'all actor rows accounted')
    return out

def compute(groups,branches,ids):
    # Actor order key is g*8+i, not the shuffled raw prompt ordinal.
    def margin(row,old,new):
        return row[ids[new]]-row[ids[old]]
    def moved(r,kind,tgt):
        i,slot,gold,other,new,t,d=r; m=branches[(t if tgt else d)+'_'+kind][i]
        return m
    def gain(r,kind,tgt):
        i,_,gold,other,new,_,_=r; old=gold if tgt else other
        return margin(moved(r,kind,tgt),old,new)-margin(branches['base'][i],old,new)
    out={}; effect={}
    for slot in ('all','first','second'):
        units=[[r for r in g.values() if slot=='all' or r[1]==slot] for g in groups]
        rows=[r for unit in units for r in unit]; summary={}
        for branch in KINDS:
            agg=defaultdict(list);counts=Counter()
            for r in rows:
                i,_,gold,other,new,_,_=r; base=branches['base'][i]
                ta=moved(r,branch,True); de=moved(r,branch,False)
                target=strict(ta,ids[new]); pre_target=strict(base,ids[new]); disrupt=strict(de,ids[gold]); pre_disrupt=strict(base,ids[gold])
                counts.update(target_worlds=1,distractor_worlds=1,base_correct_original_gold=pre_disrupt,
                    base_correct_donor_gold=pre_target,target_correct_donor_gold=target,
                    donor_gold_rescues=target and not pre_target,donor_gold_damages=pre_target and not target,
                    distractor_correct_original_gold=disrupt,distractor_harms=pre_disrupt and not disrupt,
                    distractor_restorations=disrupt and not pre_disrupt,
                    distractor_inappropriately_selects_insert=strict(de,ids[new]),
                    target_insert_beats_old=ta[ids[new]]>ta[ids[gold]],
                    distractor_original_beats_insert=de[ids[gold]]>de[ids[new]])
                for k,is_target in [('target_margin_gain',True),('distractor_margin_gain',False)]:agg[k].append(gain(r,branch,is_target))
            summary[branch]=dict(counts,**{k:mean(v) for k,v in agg.items()})
            summary[branch]['net_donor_gold_rescues']=counts['donor_gold_rescues']-counts['donor_gold_damages']
        pair_summary={}
        for k in ('native','gate','taps','joint','interaction'):
            d=[]
            def x(r,q):
                if k=='interaction':
                    return gain(r,'joint',q)-gain(r,'gate',q)-gain(r,'taps',q)
                return gain(r,k,q)
            for group in groups:
                loc=[]
                for (s,o,q),r in group.items():
                    if slot!='all' and slot!=r[1]:continue
                    sibling=group[(1-s,o,q)]
                    need(r[2]==sibling[3] and r[4]==sibling[4] and r[1]==sibling[1],
                         'crossed exact ordered edit IDs and fixed query role')
                    loc.append(x(r,True)-x(sibling,False))
                d.append(mean(loc))
            pair_summary[k]=dict(effect_nats=mean(d),group_bootstrap95_nats=ci(d),group_values=d)
        out[slot]=dict(summary=summary,crossed_exact_edit_margin_transfer=pair_summary)
    return out
def score(pin):
    led=read(P/'ledger.json',LEDGER);s=read(ROOT/'INFO002_SEAL.json',pin)
    need(s['ledger_sha256']==LEDGER and s['registered_protocol_sha256']==PROTOCOL and set(s['runs'])==set(led)==ROSTER,'sealed list')
    groups=panel();out={}
    for k,e in s['runs'].items():
        d=R/e['run_id'];r=read(d/'run.json',e['run_sha256']);a=e['artifacts']
        need(verify_run(d)['valid'] and r['status']=='succeeded' and r['artifacts']==a,'native seal')
        story=read(d/'output/intervention_receipt.json',a['intervention_receipt.json'])
        check_story(story,r,led[k],a)
        raw=bytes_at(d/'output/LOGITS.f32le',a['LOGITS.f32le'])
        need(len(raw)==9*256*ONE.size,'all logit bytes')
        wire=list(ONE.iter_unpack(raw));need(all(isfinite(v) for row in wire for v in row),'finite scores')
        branches={n:wire[i*256:(i+1)*256] for i,n in enumerate(BR)}
        ids={n:i+262 for i,n in enumerate(story['tokenizer_words'])}
        out[k]=compute(groups,branches,ids)
    a=out['info002-1371']['second'];m=a['crossed_exact_edit_margin_transfer']['taps'];c=a['summary']['taps']
    primary=dict(checkpoint='info002-1371',query_slot='second',phase='tap_only',effect_nats=m['effect_nats'],
       bootstrap95_nats=m['group_bootstrap95_nats'],sealed_baseline_vs_hybrid_gold_net_rescues=c['net_donor_gold_rescues'],
       primary_pass=m['group_bootstrap95_nats'][0]>0 and c['net_donor_gold_rescues']>0)
    print(write(ROOT/'INFO002_RESULTS.json',dict(trace_seal_sha256=pin,primary=primary,results=out,
      model_invocations_by_scorer=0,optimizer_updates=0,repair_or_language_qualification=False)))
    print(json.dumps(primary,sort_keys=True))
if __name__=='__main__':
    if sys.argv[1:] == ['seal']:seal()
    elif len(sys.argv)==3 and sys.argv[1]=='score':score(sys.argv[2])
    else:raise ValueError('seal or score SEAL_SHA256')
