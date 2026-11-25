"""Value-repair model-free verifier/scorer. Only an explicit sealed CLI may read real data."""
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import product
from math import isfinite
from pathlib import Path
from random import Random
from statistics import mean
from struct import Struct
import json,sys
from fit_value_path import ARMS,BASE,SEEDS,ENDPOINTS,need,save

FMT=Struct('<287f')
PANELS={'edits':768,'ordinary':960}
BRANCHES={('native',1371,0)}|set(product(ARMS,SEEDS,ENDPOINTS))
PHYSICAL=('first','second')
TRIPLE=('base','target','distractor')
FIELDS=('base','target','preserved','complete','distractor_insert','base_to_edit_harm','base_to_edit_restoration',
        'target_insert_beats_old','preserved_beats_insert','complete_given_base',
        'conditional_denominator_base_correct')
PAIR_FIELDS=('target','preserved','complete','distractor_insert','target_only','preserved_only','neither')
FOUR_FIELDS=('both_complete','first_only','second_only','neither','both_targets','both_preserved','either_wrong_insert')
OBSERVED_METRICS={'prediction_rows':1728,'model_forwards':216,'optimizer_updates':0}

def pinned(p,h,decode=True):
    need(p.is_file() and not p.is_symlink(),'expected regular input')
    b=p.read_bytes();need(sha256(b).hexdigest()==h,'input digest '+p.name)
    return json.loads(b) if decode else b

def bytespin(p):return sha256(p.read_bytes()).hexdigest()

def tally(rows,fields):
    return {k:sum(e[k] for e in rows) for k in fields}

def transitions(a,b,field):
    n=Counter((x[field],y[field]) for x,y in zip(a,b,strict=True))
    need(sum(n.values())==len(a),'paired denominator')
    return dict(n=len(a),gain=n[False,True],loss=n[True,False],both=n[True,True],neither=n[False,False],
                delta=n[False,True]-n[True,False])

def group_interval(ref,new,field):
    need(len(ref)==len(new) and len(ref)>0,'paired blocks')
    d=[(sum(int(y[field])-int(x[field]) for x,y in zip(a,b,strict=True)),len(a))
       for a,b in zip(ref,new,strict=True) if a]
    need(len(d)==len(ref),'empty block')
    def weighted(z):return sum(x for x,_ in z)/sum(n for _,n in z)
    r=Random(20261203); s=sorted(weighted(r.choices(d,k=len(d))) for _ in range(2000))
    return dict(mean=weighted(d),ci95=[s[50],s[1949]],blocks=len(d))

def strict(v,k):
    score=v[k]
    return score==max(v) and v.count(score)==1

def parse_judge(doc,kind,actor_sha):
    is_edit=kind=='edits'
    need(set(doc)=={'schema','actor_sha256','groups'} and doc['actor_sha256']==actor_sha and
         doc['schema']==('value-repair.edit-judge.v1' if is_edit else 'value-repair.ordinary-judge.v1'),
         'judge schema and actor digest')
    groups=doc['groups'];seen=set();all_names=set()
    need(len(groups)==(32 if is_edit else 120),'group count')
    for i,g in enumerate(groups):
        need(set(g)=={'group','name_pair','colors','cells'} and g['group']==i and
             len(g['name_pair'])==len(set(g['name_pair']))==2 and
             len(g['colors'])==len(set(g['colors']))==(3 if is_edit else 2) and len(g['cells'])==8,
             'family truth shape')
        if is_edit:
            need(tuple(g['name_pair']) not in all_names, 'edit group unique pair')
        all_names.add(tuple(g['name_pair']));keyed={}
        for c in g['cells']:
            axes=('assignment','order','query_index')
            k=tuple(c[s] for s in axes)
            need(all(type(x)is int and x in (0,1) for x in k) and k not in keyed and
                 type(c['first'])is bool and c['first']==(k[1]==k[2]),'factorial roles')
            keyed[k]=c
            common={'first',*axes}
            if is_edit:
                need(set(c)==common|{'rows','answers','inserted'} and set(c['rows'])==set(c['answers'])==set(TRIPLE)
                     and c['answers']['target']==c['inserted'] and
                     c['answers']['base']==c['answers']['distractor']!=c['inserted']
                     and set(c['answers'].values())<=set(g['colors']), 'strict edit semantics')
                rr=list(c['rows'].values())
            else:
                need(set(c)==common|{'row','answer'} and c['answer'] in g['colors'],'ordinary semantics')
                rr=[c['row']]
            need(all(type(x)is int and 0<=x<PANELS[kind] for x in rr) and len(rr)==len(set(rr))
                 and not seen.intersection(rr),'unique actor row indices')
            seen.update(rr)
        need(set(keyed)==set(product((0,1),repeat=3)),'factorial roster')
        if is_edit:
            need(len({x['inserted'] for x in keyed.values()})==1,'fixed group insertion')
        else:
            need(len({(tuple(g['name_pair']),tuple(g['colors'])) for g in groups})==len(groups),'ordinary keys disjoint')
        for (s,o,q),c in keyed.items():
            v=lambda d:(d['answers']['base'] if is_edit else d['answer'])
            need(v(c)==v(keyed[s,1-o,q]) and v(c)!=v(keyed[s,o,1-q]) and
                 v(c)==v(keyed[1-s,o,1-q]), 'fixed rendered assignment and opposite-query answer')
    need(seen==set(range(PANELS[kind])),'full actor index roster')
    return groups

def syntax_audit(actor,groups,kind):
    need(set(actor)=={'schema','prompts'} and actor['schema']=='value-repair.blind-prompts.v1'
         and len(actor['prompts'])==len(set(actor['prompts']))==PANELS[kind],'opaque actor roster')
    def analyze(row):
        raw=actor['prompts'][row];t=tuple(raw.split())
        need(len(t)==7 and ' '.join(t)==raw and t[1]==t[4]=='is' and
             t[0]!=t[3] and t[2]!=t[5] and t[6] in (t[0],t[3]),'grammar ceiling input')
        at=2 if t[6]==t[0] else 5
        return t,at,t[at]
    seen=0
    for g in groups:
        cells={}
        for c in g['cells']:
            s,o,q=[c[k] for k in ('assignment','order','query_index')]
            if kind=='edits':
                a,b,d=(analyze(c['rows'][k]) for k in TRIPLE)
                h,j,_=a
                need(j==b[1]==d[1] and [i for i,(x,y) in enumerate(zip(h,b[0],strict=True)) if x!=y]==[j]
                     and [i for i,(x,y) in enumerate(zip(h,d[0],strict=True)) if x!=y]==[7-j]
                     and (a[2],b[2],d[2])==tuple(c['answers'][k] for k in TRIPLE)
                     and b[0][j]==d[0][7-j]==c['inserted'],'same edit and exact grammar ceiling')
                pack=(a[0],b[0],d[0]);seen+=3
            else:
                h,j,y=analyze(c['row'])
                need(y==c['answer'],'ordinary exact grammar ceiling')
                pack=(h,);seen+=1
            np=g['name_pair']
            need((h[0],h[3],h[6])==(np[o],np[1-o],np[q]) and (j==2)==c['first'],
                 'literal role vs judge')
            cells[s,o,q]=pack
        for s,o in product((0,1),repeat=2):
            a,b=cells[s,o,o],cells[s,o,1-o]
            need(a[0][:-1]==b[0][:-1],'identical pre-edit history')
            if kind=='edits':
                need(a[1][:-1]==b[2][:-1] and b[1][:-1]==a[2][:-1],'same physical edit across queries')
    need(seen==PANELS[kind],'engineered ceiling denominator')
    return dict(exact_syntax_answers=seen,rows=seen,learned=False)

def evaluate_edits(rows,groups,ids):
    t={k:[] for k in ('first','second','all')}
    pairs={k:[] for k in ('edit_first','edit_second','all','four')}
    for g in groups:
        out={};by={}
        for c in g['cells']:
            z={k:rows[i] for k,i in c['rows'].items()};ys={k:ids[w] for k,w in c['answers'].items()}
            b,ta,pr=(strict(z[k],ys[k]) for k in TRIPLE)
            ins=strict(z['distractor'],ys['target'])
            out[c['assignment'],c['order'],c['query_index']]=dict(zip(FIELDS,
                (b,ta,pr,b and ta and pr,ins,b and not pr,pr and not b,
                 z['target'][ys['target']]>z['target'][ys['base']],
                 z['distractor'][ys['base']]>z['distractor'][ys['target']],b and ta and pr,b),strict=True))
            by[c['assignment'],c['order'],c['query_index']]=c['first']
        t['all'].append(list(out.values()))
        for k,slot in zip(PHYSICAL,(True,False),strict=True):
            t[k].append([v for key,v in out.items() if by[key]==slot])
        e1=[];e2=[];four=[]
        def pair(a,other):
            ta=a['target']; pr=other['preserved']
            return dict(zip(PAIR_FIELDS,(ta,pr,ta and pr,other['distractor_insert'],
                                         ta and not pr,pr and not ta,not ta and not pr),strict=True))
        for s,o in product((0,1),repeat=2):
            a,b=out[s,o,o],out[s,o,1-o]
            u,v=pair(a,b),pair(b,a)
            e1.append(u);e2.append(v)
            x,y=u['complete'],v['complete']
            four.append(dict(zip(FOUR_FIELDS,(x and y,x and not y,y and not x,not x and not y,
                u['target'] and v['target'],u['preserved'] and v['preserved'],
                u['distractor_insert'] or v['distractor_insert']),strict=True)))
        pairs['edit_first'].append(e1);pairs['edit_second'].append(e2)
        pairs['all'].append(e1+e2);pairs['four'].append(four)
    return t,pairs

def evaluate_ordinary(rows,groups,ids):
    cells=[];fams=defaultdict(list)
    for g in groups:
        b=[dict(correct=strict(rows[c['row']],ids[c['answer']]),first=c['first']) for c in g['cells']]
        cells.append(b);fams[tuple(g['name_pair'])].append(dict(complete=all(x['correct'] for x in b)))
    fams=[fams[k] for k in sorted(fams)]
    return cells,fams

def flat(g):return [x for y in g for x in y]

def stats(v):
    t,p,(cells,fams)=v
    def summary(groups,fields):
        a=flat(groups);return dict(n=len(a),**tally(a,fields))
    tri={k:summary(b,FIELDS) for k,b in t.items()}
    for v in tri.values():
        v['complete_given_base_rate']=v['complete_given_base']/v['conditional_denominator_base_correct'] if v['conditional_denominator_base_correct'] else None
    return dict(triplets=tri,opposite_query={k:summary(v,FOUR_FIELDS if k=='four' else PAIR_FIELDS) for k,v in p.items()},
                ordinary=dict(n=len(flat(cells)),worlds_correct=sum(x['correct'] for b in cells for x in b),
                     first_correct=sum(x['first'] and x['correct'] for b in cells for x in b),
                     second_correct=sum(not x['first'] and x['correct'] for b in cells for x in b),
                     complete_families=sum(x['complete'] for x in flat(fams)),families=len(flat(fams))))

def paired(ref,new):
    a,p,(ac,af)=ref; b,q,(bc,bf)=new
    ordinary={tag:transitions([x for x in flat(ac) if flag is None or x['first']==flag],
       [x for x in flat(bc) if flag is None or x['first']==flag],'correct') for tag,flag in (('all',None),('first',True),('second',False))}
    ordinary['families']=transitions(flat(af),flat(bf),'complete')
    ordinary['family_interval']=group_interval(af,bf,'complete')
    trip={}
    for k in a:
        z={field:transitions(flat(a[k]),flat(b[k]),field) for field in ('base','target','preserved','complete','distractor_insert')}
        z['complete_interval']=group_interval(a[k],b[k],'complete')
        trip[k]=z
    opp={k:transitions(flat(p[k]),flat(q[k]),'both_complete' if k=='four' else 'complete') for k in p}
    return dict(triplets=trip,opposite_query=opp,ordinary=ordinary)

def gate(native,control,content):
    n,c,t=[v['triplets']['second'] for v in (native,control,content)]
    obj=content['ordinary']
    need(n['n']==c['n']==t['n']==128 and obj['n']==960 and obj['families']==120,'locked denominators')
    gain=t['complete']-c['complete']
    first_loss=native['ordinary']['first_correct']-obj['first_correct']
    conditions=dict(second_triplet_gain_at_least_7=gain>=7,
      second_distractor_no_increase_vs_control=t['distractor_insert']<=c['distractor_insert'],
      second_distractor_no_increase_vs_native=t['distractor_insert']<=n['distractor_insert'],
      second_preserved_no_decrease_vs_control=t['preserved']>=c['preserved'],
      second_preserved_no_decrease_vs_native=t['preserved']>=n['preserved'],
      ordinary_912=obj['worlds_correct']>=912,ordinary_108=obj['complete_families']>=108,
      ordinary_first_loss_at_most_5=first_loss<=5)
    return dict(conditions=conditions,pass_all=all(conditions.values()),second_triplet_gain=gain,ordinary_first_loss=first_loss)

def decode_wire(p,h,n):
    b=pinned(p,h,False);need(len(b)==n*FMT.size,'full binary shape')
    z=list(FMT.iter_unpack(b));need(all(isfinite(x) for r in z for x in r),'nonfinite logits')
    return z

def score(root,seal,review_plan):
    from trainlab.runner import verify_run
    roster={}
    need(set(seal['actors'])==set(review_plan['actors']) and len(seal['actors'])==13,'exact sealed roster')
    values={}
    judge={};oracle={}
    for j,n in PANELS.items():
        e=review_plan['judge'][j]
        judge[j]=parse_judge(pinned(root/'realized-data-v1/held_judge'/f'{j}.json',e['judge_sha256']),j,e['actor_sha256'])
        raw=pinned(root/'realized-data-v1/held_actor'/f'{j}.json',e['actor_sha256'])
        oracle[j]=syntax_audit(raw,judge[j],j)
    words=None
    for label,s in sorted(seal['actors'].items()):
        p=review_plan['actors'][label]; k=tuple(p[f] for f in ('arm','seed','updates'))
        need(k in BRANCHES and k not in roster and Path(s['run_id']).name==s['run_id'],'branch uniqueness')
        run=root/'value_eval_runs_v3'/s['run_id']; o=run/'output'
        native=pinned(run/'run.json',s['run_sha256'])
        need(verify_run(run)['valid'] and native['name']==label and native['status']=='succeeded' and native['exit_code']==0
          and native['elapsed_seconds']<=45 and native['metrics']==OBSERVED_METRICS and native['inputs']==p['inputs']
          and native['artifacts']==s['artifacts'] and native['seed']==k[1],'sealed native verification')
        receipt=pinned(o/'actor_receipt.json',s['artifacts']['actor_receipt.json'])
        need(receipt['schema']=='cog-info-value-repair.actor.v1' and
          tuple(receipt[f] for f in ('arm','seed','updates'))==k and receipt['backbone_sha256']==BASE
          and receipt['adapter_sha256']==p.get('adapter_sha256') and receipt['metrics']==OBSERVED_METRICS
          and receipt['native_zero_adapter_computation']==(k[0]=='native'),'frozen actor identity')
        tok=receipt['tokenizer_words']
        if words is None:words=tok
        need(tok==words and len(set(tok))==25,'consistent frozen word roster')
        ids={w:262+i for i,w in enumerate(words)}
        z={}
        for panel,n in PANELS.items():
            f=f'{panel.upper()}-LOGITS.f32le'; h=s['artifacts'][f]
            need(receipt['panels'][panel]==dict(actor_sha256=review_plan['judge'][panel]['actor_sha256'],
                logits_sha256=h,shape=[n,287]),'actor/judge/array binding')
            z[panel]=decode_wire(o/f,h,n)
        values[k]=(*evaluate_edits(z['edits'],judge['edits'],ids),
                   evaluate_ordinary(z['ordinary'],judge['ordinary'],ids));roster[k]=label
    need(set(roster)==BRANCHES,'13 fixed branches')
    native=('native',1371,0); results={k:stats(v) for k,v in values.items()}
    variants={roster[k]:dict(summary=results[k],paired_vs_native=paired(values[native],v))
              for k,v in values.items() if k!=native}
    comparisons={}
    for seed,step in product(SEEDS,ENDPOINTS):
        c=('literal_lag',seed,step);t=('tap_content',seed,step)
        comparisons[f'{seed}-{step}']=dict(seed=seed,updates=step,descriptive_only=step!=1280,
            paired=paired(values[c],values[t]), fixed_thresholds=gate(results[native],results[c],results[t]))
    all_pass=all(comparisons[f'{seed}-1280']['fixed_thresholds']['pass_all'] for seed in SEEDS)
    return dict(status='LOCAL_SYNTHETIC_NO_PROMOTION',native=results[native],
       variants=variants,matched_comparisons=comparisons,every_final_seed_met_quality_gates=all_pass,
       engineered_grammar_ceiling=oracle,
       promoted=False,model_invocations=0,optimizer_updates=0)

if __name__=='__main__':
    root=Path(__file__).resolve().parent
    need(len(sys.argv)==4,'usage: score_value_path_v3.py PLAN_SHA SEAL_SHA EXACT_SCORE_GO_SHA')
    plan_sha,seal_sha,go_sha=sys.argv[1:]
    go=pinned(root/'VALUE_EVAL_SCORE_GO_V3.json',go_sha)
    need(go==dict(status='GO_SCORE_ONLY',plan_sha256=plan_sha,seal_sha256=seal_sha,
                  scorer_sha256=bytespin(Path(__file__))), 'exact independent separate score GO')
    plan=pinned(root/'VALUE_EVAL_PLAN_V3.json',plan_sha)
    seal=pinned(root/'VALUE_EVAL_SEAL_V3.json',seal_sha)
    need(seal['plan_sha256']==plan_sha and
         plan['status']=='AWAITING_EXACT_V3_HELD_ACTOR_ONLY_DISPATCH' and
         plan['source_sha256']['score_value_path_v3.py']==bytespin(Path(__file__)) and
         seal['train_dry_seal_sha256']==plan['train_dry_seal_sha256'],'sealed v3 plan and frozen scorer')
    out=root/'VALUE_EVAL_RESULTS_V3.json'
    result=score(root,seal,plan)
    save(out,dict(result,eval_plan_sha256=plan_sha,eval_seal_sha256=seal_sha,eval_score_go_sha256=go_sha,
                  eval_scorer_sha256=bytespin(Path(__file__))))
    print(bytespin(out))
