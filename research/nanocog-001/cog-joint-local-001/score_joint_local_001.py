"""Independent label-side exact JOINT scoring, no Torch or model invocation."""
from collections import Counter,defaultdict
from hashlib import sha256
from itertools import product
from math import isfinite
from pathlib import Path
from random import Random
from struct import Struct
import json,sys
from joint_local_objective import ARMS,SEEDS,ENDPOINTS,need
from joint_local_intake import parse_prompt,digest
from joint_eval_checkpoint import WORDS,NATIVE_FILE

ROSTER={('native',1371,0)}|set(product(ARMS,SEEDS,ENDPOINTS))
ROLES=('base_first','base_second','edit_first_first','edit_first_second','edit_second_first','edit_second_second')
ROWS={'edits':768,'ordinary':960}
EXPECTED=dict(prediction_rows=1728,model_forwards=216,optimizer_updates=0)
ARTIFACTS={'actor_receipt.json','EDITS-LOGITS.f32le','ORDINARY-LOGITS.f32le'}
IDS={w:262+i for i,w in enumerate(WORDS)}
WIRE=Struct('<287f')
ROLE_TRIPLE={'first':(0,2,4),'second':(1,5,3)}
PAIR={'first':(2,3),'second':(5,4)}
BSEED=20261204
R=Path(__file__).resolve().parent

def read(path,h,*,raw=False):
    need(path.is_file() and not path.is_symlink(),'regular verified artifact')
    b=path.read_bytes();need(sha256(b).hexdigest()==h,'pinned bytes: '+path.name)
    return b if raw else json.loads(b)

def write(path,v):
    with path.open('x') as f:json.dump(v,f,sort_keys=True,allow_nan=False,separators=(',',':'));f.write('\n')

def strict(row,ans):
    z=row[ans];return max(row)==z and row.count(z)==1

def prompt_projection(d,n):
    need(set(d)=={'schema','prompts'} and d['schema']=='joint-local.actor.v1' and
         len(d['prompts'])==len(set(d['prompts']))==n,'opaque actor scope')
    p=[parse_prompt(x) for x in d['prompts']]
    need(all(w in IDS for x in p for w in x),'literal tokens in vocabulary')
    return p

def exact_answer(t):return t[2 if t[6]==t[0] else 5]

def edit_groups(actor,judge,actor_sha):
    p=prompt_projection(actor,768)
    need(set(judge)=={'schema','groups','actor_sha256'} and judge['schema']=='joint-local.edit-judge.v1'
         and judge['actor_sha256']==actor_sha and len(judge['groups'])==32,'edit judge schema')
    groups=[];seen=set();pairs=set()
    for i,g in enumerate(judge['groups']):
        names,colors=tuple(g['name_pair']),tuple(g['colors'])
        need(set(g)=={'name_pair','colors','group','histories'} and g['group']==i and
             len(names)==len(set(names))==2 and len(colors)==len(set(colors))==3 and names not in pairs and
             len(g['histories'])==4,'exact edit block family')
        pairs.add(names);ss={}; insertion=set();group=[]
        for h in g['histories']:
            need(set(h)=={'assignment','order','rows','answers','inserted','query_rows','physical_rows'} and
                 set(h['rows'])==set(h['answers'])==set(ROLES),'exact history judge wire')
            rows=tuple(h['rows'][k] for k in ROLES); mr=h['inserted']; insertion.add(mr)
            need(all(type(k)is int and 0<=k<768 for k in rows) and len(set(rows))==6 and seen.isdisjoint(rows),
                 'unique six factor rows')
            seen.update(rows); t=[p[j] for j in rows]; yy=tuple(h['answers'][k] for k in ROLES);base=t[0][:-1]
            need(tuple(map(exact_answer,t))==yy and set(yy)==set(colors) and yy[2]==yy[5]==mr and
                 yy[0]==yy[4] and yy[1]==yy[3] and mr not in yy[:2],'engineered syntax matches independent labels')
            for j in range(0,6,2):
                a,b=t[j:j+2]
                need(a[:-1]==b[:-1] and (a[6],b[6])==(base[0],base[3]),'opposite queries fixed history')
            dif=lambda x:[j for j,(a,b) in enumerate(zip(base,x[:-1],strict=True)) if a!=b]
            need(dif(t[2])==[2] and dif(t[4])==[5],'identical physical edits across query pair')
            axis=(h['assignment'],h['order'])
            need(all(type(v)is int and v in (0,1) for v in axis) and axis not in ss and
                 (base[0],base[3])==(names[axis[1]],names[1-axis[1]]),'assignment/order factorial')
            ss[axis]=(yy[0],yy[1])
            need(h['query_rows']=={k:dict(zip(('base','target','distractor'),(rows[j] for j in v),strict=True))
                    for k,v in ROLE_TRIPLE.items()} and
                 h['physical_rows']=={'first':[rows[2],rows[3]],'second':[rows[4],rows[5]]},
                 'derived labelled row joins reconcile')
            group.append((rows,tuple(IDS[x] for x in yy),IDS[mr]))
        need(set(ss)==set(product((0,1),repeat=2)) and len(insertion)==1 and
             len(set(ss.values()))==2 and all(ss[s,1-o]==ss[1-s,o]==ss[s,o][::-1] for s,o in ss),
             'both assignments, both orders, fixed group insertion')
        groups.append(group)
    need(seen==set(range(768)),'exact roster coverage')
    return groups

def ordinary_groups(actor,judge,actor_sha):
    p=prompt_projection(actor,960)
    need(set(judge)=={'schema','groups','actor_sha256'} and judge['schema']=='joint-local.ordinary-judge.v1'
         and judge['actor_sha256']==actor_sha and len(judge['groups'])==120,'ordinary judge schema')
    out=[];seen=set();keys=set()
    for i,g in enumerate(judge['groups']):
        names,colors=tuple(g['name_pair']),tuple(g['colors'])
        need(set(g)=={'name_pair','colors','group','rows'} and g['group']==i and
             len(names)==len(set(names))==len(colors)==len(set(colors))==2 and (names,colors) not in keys and
             len(g['rows'])==8,'exact ordinary family')
        keys.add((names,colors));group=[];cells=set()
        for x in g['rows']:
            r=x['row'];need(set(x)=={'row','answer','query_in_first'} and type(r)is int and 0<=r<960 and r not in seen,
                      'unique ordinary rows')
            seen.add(r);a=p[r];h=exact_answer(a);first=a[6]==a[0]
            need({a[0],a[3]}==set(names) and {a[2],a[5]}==set(colors) and h==x['answer'] and
                 type(x['query_in_first'])is bool and first==x['query_in_first'],'ordinary role alignment')
            cells.add((a[0],a[2],a[6]));group.append((r,IDS[h],first))
        need(len(cells)==8,'complete ordinary assignment/order/query cube')
        out.append((names,group))
    need(seen==set(range(960)),'all ordinary actor rows')
    return out

def evaluate(ze,zo,edits,ordinary):
    t={k:[] for k in ('first','second','all')};pairs={k:[] for k in ('first','second','four')}
    for group in edits:
        occ={k:[] for k in ('first','second','all')};ab={k:[] for k in pairs}
        for rows,y,ins in group:
            z=[ze[i] for i in rows];ok=[strict(r,g) for r,g in zip(z,y,strict=True)]
            for side,(b,v,d) in ROLE_TRIPLE.items():
                o=dict(base=ok[b],target=ok[v],preserved=ok[d],complete=ok[b] and ok[v] and ok[d],
                    distractor_insert=strict(z[d],ins),base_to_edit_harm=ok[b] and not ok[d],
                    base_to_edit_restoration=ok[d] and not ok[b],target_beats_old=z[v][ins]>z[v][y[b]],
                    preserved_beats_insert=z[d][y[b]]>z[d][ins])
                occ[side].append(o);occ['all'].append(o)
            complete=[]
            for side,(v,d) in PAIR.items():
                both=ok[v] and ok[d];complete.append(both)
                ab[side].append(dict(complete=both,target=ok[v],preserved=ok[d],wrong_insert=strict(z[d],ins),
                                     target_only=ok[v] and not ok[d],preserved_only=ok[d] and not ok[v]))
            ab['four'].append(dict(complete=all(complete),first_only=complete[0] and not complete[1],
                         second_only=complete[1] and not complete[0],neither=not any(complete)))
        for k in t:t[k].append(occ[k])
        for k in pairs:pairs[k].append(ab[k])
    cells=defaultdict(list); fams=defaultdict(list)
    for names,group in ordinary:
        rr=[dict(correct=strict(zo[r],gold),first=first) for r,gold,first in group]
        cells[names].extend(rr);fams[names].append(dict(complete=all(c['correct'] for c in rr)))
    return t,pairs,[cells[k] for k in sorted(cells)],[fams[k] for k in sorted(fams)]

def flat(x):return [r for g in x for r in g]

def tally(x):
    r=flat(x);return dict(n=len(r),**{k:sum(q[k] for q in r) for k in r[0]})

def summary(v):
    t,p,cc,ff=v; c=flat(cc)
    tri={k:tally(g) for k,g in t.items()}
    for g in tri.values():g['complete_given_base']=g['complete']/g['base'] if g['base'] else None
    return dict(triplets=tri,opposite={k:tally(g) for k,g in p.items()},
        ordinary=dict(n=len(c),worlds_correct=sum(x['correct'] for x in c),families=len(flat(ff)),
          families_complete=sum(x['complete'] for x in flat(ff)),first_correct=sum(x['first'] and x['correct'] for x in c),
          second_correct=sum(not x['first'] and x['correct'] for x in c)))

def comparison(ref,after,k):
    pair=Counter((a[k],b[k]) for a,b in zip(flat(ref),flat(after),strict=True))
    d=[(sum(int(b[k])-int(a[k]) for a,b in zip(x,y,strict=True)),len(x)) for x,y in zip(ref,after,strict=True)]
    need(all(n for _,n in d),'paired named blocks')
    rng=Random(BSEED)
    ratio=lambda rows:sum(u for u,_ in rows)/sum(n for _,n in rows)
    ci=sorted(ratio(rng.choices(d,k=len(d))) for _ in range(2000))
    return dict(n=sum(pair.values()),gain=pair[False,True],loss=pair[True,False],
        both=pair[True,True],neither=pair[False,False],delta=pair[False,True]-pair[True,False],
        blocked_mean=ratio(d),blocked_ci95=[ci[50],ci[1949]],blocks=len(d))

def paired(ref,after):
    a,b,o,f=ref;x,y,p,g=after
    return dict(triplets={k:{q:comparison(a[k],x[k],q) for q in
                        ('base','target','preserved','complete','distractor_insert')} for k in a},
        opposite={k:comparison(b[k],y[k],'complete') for k in b},
        ordinary={tag:comparison([[r for r in b if flag is None or r['first']==flag] for b in o],
                                  [[r for r in b if flag is None or r['first']==flag] for b in p],'correct')
                 for tag,flag in (('overall',None),('first',True),('second',False))}|
                 {'families':comparison(f,g,'complete')})

def absolute(ref,new):
    a=ref['triplets'];b=new['triplets'];n=ref['ordinary'];o=new['ordinary']
    need(all(a[k]['n']==b[k]['n']==128 for k in ('first','second')) and o['n']==960 and o['families']==120,
         'original strict denominators')
    out=dict(ordinary_912=o['worlds_correct']>=912,ordinary_108=o['families_complete']>=108,
             ordinary_first_loss_at_most_5=n['first_correct']-o['first_correct']<=5)
    for k in ('first','second'):
        out.update({f'{k}_116':b[k]['complete']>=116,
                    f'{k}_no_excess_insert':b[k]['distractor_insert']<=a[k]['distractor_insert'],
                    f'{k}_no_preservation_decrease':b[k]['preserved']>=a[k]['preserved']})
    out.update({f'{k}_same_or_better_vs_native':new['opposite'][k]['complete']>=ref['opposite'][k]['complete'] for k in ('first','second','four')})
    return dict(pass_all=all(out.values()),criteria=out)

def incremental(native,mean,tail):
    n=mean['triplets'];t=tail['triplets']
    gain=t['second']['complete']-n['second']['complete']
    o=tail['ordinary']
    out=dict(second_plus_7=gain>=7,ordinary_912=o['worlds_correct']>=912,ordinary_108=o['families_complete']>=108,
        first_ordinary_loss_vs_native_at_most_5=native['ordinary']['first_correct']-o['first_correct']<=5,
        first_ordinary_loss_vs_mean_at_most_5=mean['ordinary']['first_correct']-o['first_correct']<=5)
    for name,ref,roles in (('native',native,('second',)),('matched',mean,('first','second'))):
        for k in roles:
            b=ref['triplets'][k]
            out.update({f'{k}_no_excess_insert_vs_{name}':t[k]['distractor_insert']<=b['distractor_insert'],
                        f'{k}_no_preservation_decrease_vs_{name}':t[k]['preserved']>=b['preserved']})
        out.update({f'{k}_same_or_better_vs_{name}':tail['opposite'][k]['complete']>=ref['opposite'][k]['complete']
                    for k in ('first','second','four')})
    return dict(pass_all=all(out.values()),second_gain=gain,primary_ceiling_limited=n['second']['complete']>121,criteria=out)

def derive_decision(abs_pass,inc,train_ok):
    m,t,a=(abs_pass[k] for k in ('gated32_mean','gated32_tail','attention32_tail'))
    adequate=all(train_ok.values())
    if t and inc:
        route='GATED_TAIL_WINS_FIXED_ABSOLUTE_AND_MATCHED_INCREMENTAL'
    elif m:
        route='GATED_MEAN_ABSOLUTE_PREFER_SIMPLE'
    elif a and not t and adequate:
        route='BOUNDED_MIXER_GAP_THIS_ADEQUATE_FIXED_REGIME'
    elif not adequate:
        route='UNDERTRAINED_NO_SCAFFOLD_ATTRIBUTION'
    elif t:
        route='GATED_TAIL_ABSOLUTE_NO_REGISTERED_INCREMENTAL_ADVANCEMENT'
    else:
        route='NO_ALL_SEED_CAPABILITY_AT_FIXED_BUDGET'
    return dict(absolute_all_seeds=abs_pass,incremental_all_seeds=inc,all_fit_audits_adequate=adequate,
                bounded_scaffold_gap=a and not m and not t and adequate,disposition=route,
                promoted=False,serving_cost_not_established=True)

def decode_wire(path,h,n):
    b=read(path,h,raw=True);need(len(b)==n*WIRE.size,'exact 287-f32 wire length')
    z=list(WIRE.iter_unpack(b))
    need(all(isfinite(x) for row in z for x in row),'finite logits')
    return z

def score(root,plan,seal):
    from trainlab.runner import verify_run
    need(set(plan['actors'])==set(seal['runs']) and len(seal['runs'])==19,'complete 19 actor seal')
    g=root/'joint-realized-data-v1'
    def data(panel,mode):
        pin=plan['panels'][panel][mode+'_sha256']
        return read(g/f'held_{mode}'/f'{panel}.json',pin)
    ed=edit_groups(data('edits','actor'),data('edits','judge'),plan['panels']['edits']['actor_sha256'])
    ordinary=ordinary_groups(data('ordinary','actor'),data('ordinary','judge'),plan['panels']['ordinary']['actor_sha256'])
    need({x[0] for x in ordinary}.isdisjoint({tuple(g['name_pair']) for g in data('edits','judge')['groups']}),
         'held name-pair separation')
    vals={};tags={};fit_ok={}
    for n,s in sorted(seal['runs'].items()):
        ref=plan['actors'][n]; check=ref['checkpoint']
        k=tuple(check[f] for f in ('arm','seed','updates'))
        need(k in ROSTER and k not in vals and Path(s['run_id']).name==s['run_id'],'fixed scenario roster')
        native=root/'joint_eval_runs'/s['run_id']
        run=read(native/'run.json',s['run_sha256']);out=native/'output'
        need(verify_run(native)['valid'] and run['name']==n and run['status']=='succeeded' and run['exit_code']==0
             and 0<run['elapsed_seconds']<=45 and run['inputs']==ref['inputs'] and run['artifacts']==s['artifacts']
             and run['metrics']==EXPECTED and set(run['artifacts'])==ARTIFACTS,'exact score-only sealed actors')
        a=read(out/'actor_receipt.json',s['artifacts']['actor_receipt.json'])
        need(a['schema']=='cog-joint-local-001.blind-native.v1' and
             tuple(a[f] for f in ('arm','seed','updates'))==k and a['checkpoint_sha256']==check['sha256'] and
             a['state_sha256']==check['state_sha256'] and a['tokenizer_words']==list(WORDS) and
             a['mixer']==('gated32' if k[0]=='native' else ARMS[k[0]][0]) and a['metrics']==EXPECTED,
             'complete exact actor sidecar')
        rows={}
        for panel,size in ROWS.items():
            f=f'{panel.upper()}-LOGITS.f32le';pin=s['artifacts'][f]
            need(a['panels'][panel]==dict(shape=[size,287],actor_sha256=plan['panels'][panel]['actor_sha256'],
                                         logits_sha256=pin),'exact prediction to truth row binding')
            rows[panel]=decode_wire(out/f,pin,size)
        vals[k]=evaluate(rows['edits'],rows['ordinary'],ed,ordinary);tags[k]=n
        if k[0]!='native':
            aok=ref['independent_train_fit_adequate']; title=f'{k[0]}-{k[1]}'
            need(type(aok)is bool and (title not in fit_ok or fit_ok[title]==aok),'one audited adequacy per trajectory')
            fit_ok[title]=aok
    need(set(vals)==ROSTER and len(fit_ok)==9,'no arm/endpoint selection')
    sums={k:summary(v) for k,v in vals.items()}
    nk=('native',1371,0);base=sums[nk]; obs={}; extra={}
    for k in sorted(vals):
        if k==nk:continue
        arm,seed,u=k
        obs[tags[k]]=dict(summary=sums[k],absolute=absolute(base,sums[k]),descriptive_early=u==128,
                          compared_to_native=paired(vals[nk],vals[k]))
    for seed,u in product(SEEDS,ENDPOINTS):
        m=('gated32_mean',seed,u);g=('gated32_tail',seed,u);a=('attention32_tail',seed,u)
        extra[f'{seed}-{u}']=dict(incremental=incremental(base,sums[m],sums[g]),
            gated_tail_vs_mean=paired(vals[m],vals[g]),attention_tail_vs_gated_tail=paired(vals[g],vals[a]),
            descriptive_early=u==128)
    all_abs={a:all(obs[tags[a,s,1280]]['absolute']['pass_all'] for s in SEEDS) for a in ARMS}
    inc=all(extra[f'{seed}-1280']['incremental']['pass_all'] for seed in SEEDS)
    return dict(native=base,predictions=obs,matched_paired_comparisons=extra,train_fit_adequacy_by_run=fit_ok,
                decision=derive_decision(all_abs,inc,fit_ok),score_model_invocations=0,optimizer_updates=0,
                engineered_ceiling=dict(correct_edit=768,correct_ordinary=960,learned=False))

if __name__=='__main__':
    need(len(sys.argv)==4,'PLAN_SHA SEAL_SHA INDEPENDENT_SCORE_GO_SHA')
    p,s,g=sys.argv[1:];go=read(R/'JOINT_EVAL_SCORE_GO.json',g)
    sh=digest(Path(__file__))
    need(go==dict(status='GO_JOINT_SCORE_ONLY',plan_sha256=p,seal_sha256=s,scorer_sha256=sh),'separate score-only authority')
    plan=read(R/'JOINT_EVAL_PLAN.json',p); seal=read(R/'JOINT_EVAL_SEAL.json',s)
    need(seal['plan_sha256']==p and plan['sources'][Path(__file__).name]==sh,'exact plan/score lineage')
    out=R/'JOINT_EVAL_RESULTS.json'
    write(out,dict(score(R,plan,seal),plan_sha256=p,seal_sha256=s,score_go_sha256=g,scorer_sha256=sh))
    print(digest(out))
