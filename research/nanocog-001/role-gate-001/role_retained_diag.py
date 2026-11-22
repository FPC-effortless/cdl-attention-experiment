"""Independent, model-free ROLE-002/003 diagnostic from sealed retained logits."""
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import product, combinations
import json
import math
from pathlib import Path
import sys
from trainlab.runner import verify_run

PINS={
'002':('201a63921dfaaf601b03e03e7dc80858e9958b71fe0eef8e6b8572d8d459aac1','ce80c30bfbb9c38ae8b078d16c0fd8187bc8a641bba4315bd05dad5253927118','2f28213c2eedcbdfce2151a42250ba57b0ca48bd92a2119726dc10818679ad25','e6a31a5ceaecb1ee296289f4f38a63d541f44de6e0508c02ad55ac9542e056f8','a6c354d69ee3ac3583cab31125404987fe510c22e3e282d93325ab0a04b390cb'),
'003':('536ba918ab5122549cd1142df0c02789fb406731a2fdf4928db75ac04a1c1555','edade4df767fff4ce0321d09c4e5511514f3944e8ea5f52077539402b7f4de84','d9b6bd3cd2a9a089933c51a14383bc3378d88e837d9d5427e81175d22a826d48','a03f10d2ec6b2ef1fee936c031dad2f17ab6d7499f61cc6fb761bb1c72321d12','8f80309fb58d1f39a7533e8aeb3b28305957c4894e0ccee154470f25fc8b43d2')}
S=(1171,1271,1371)
PAT={'binding':(0,2),'position':(0,1),'canonical_name':(0,),'color':(),
     'order_only':(1,),'query_only':(2,),'order_query':(1,2),'triple':(0,1,2)}
def need(x,msg):
    if not x: raise ValueError(msg)
def read(p,pin=None):
    b=p.read_bytes(); need(pin is None or sha256(b).hexdigest()==pin,f'hash: {p}')
    return json.loads(b)
def h(p):return sha256(p.read_bytes()).hexdigest()
def sign(x):return (x>0)-(x<0)
def coeff(c,ix):return math.prod(c[i] for i in ix)
def summarize(fs, logits, ids):
    strat=defaultdict(Counter); flips=defaultdict(Counter); hist=Counter()
    templates=Counter(); walsh=Counter(); energies=Counter()
    fam_success=[]; family_fails_by_name=defaultdict(Counter); family_fails_by_color=defaultdict(Counter)
    for key,rows in fs:
        colors=key[1]; out={}
        for c,idx,y in rows:
            zz=logits[idx]; need(len(zz)==287 and all(type(x) in (float,int) and math.isfinite(x) for x in zz),'invalid vector')
            d=zz[ids[colors[0]]]-zz[ids[colors[1]]]
            exact=all(zz[ids[y]]>v for k,v in enumerate(zz) if k!=ids[y])
            pos=c[1]*c[2]
            need(sign(1 if y==colors[0] else -1)==c[0]*c[2],'truth/factor disagreement')
            out[c]=(d,exact)
            for st in ('all',f'order={c[1]:+d},query={c[2]:+d}',f'position={1 if pos==1 else 2}',
                       f'order={c[1]:+d}',f'query={c[2]:+d}',
                       f'position={1 if pos==1 else 2},correct_color={1 if y==colors[0] else 2}'):
                strat[st].update(worlds=1,exact=exact,relative_correct=sign(d)==c[0]*c[2],
                                 chooses_first=sign(d)==c[0]*c[1],ties=not sign(d))
        n=sum(x[1] for x in out.values()); hist[n]+=1; fam_success.append(n==8)
        family_fails_by_name[key[0]].update(families=1,failed=n!=8)
        family_fails_by_color[key[1]].update(families=1,failed=n!=8)
        for tag,ix in PAT.items():
            ds=[sign(d)*coeff(c,ix) for c,(d,e) in out.items()]
            for polarity in (1,-1):
                if all(v==polarity for v in ds): templates[f'{tag}:{polarity:+d}']+=1
        cf={tag:math.fsum(d*coeff(c,ix) for c,(d,_) in out.items())/8 for tag,ix in PAT.items()}
        walsh.update(cf); den=math.fsum(v*v for v in cf.values())
        for tag,v in cf.items():
            energies[tag]+=v*v/den if den else 0
        for a,axis in enumerate(('assignment','order','query')):
            for c,(d,e) in out.items():
                if c[a]!=1:continue
                cp=c[:a]+(-1,)+c[a+1:]
                dp,ep=out[cp]
                fixed=tuple((i,c[i]) for i in range(3) if i!=a)
                for st in ('all',str(fixed)):
                    flips[(axis,st)].update(pairs=1,both_exact=e and ep,
                         both_relative_correct=sign(d)==c[0]*c[2] and sign(dp)==cp[0]*cp[2],
                         same_color_non_tied=bool(sign(d)) and sign(d)==sign(dp),
                         flipped_color_non_tied=bool(sign(d)) and sign(d)==-sign(dp),
                         one_or_more_ties=not(sign(d) and sign(dp)),
                         one_exact=e != ep,neither_exact=not(e or ep))
    ccl=lambda d:dict(groups=len(d),with_failures=sum(x['failed']>0 for x in d.values()),
                      singleton_groups=sum(x['families']==1 for x in d.values()),
                      repeated_both_failed=sum(math.comb(x['failed'],2) for x in d.values()),
                      repeated_pairs=sum(math.comb(x['families'],2) for x in d.values()))
    return dict(strata=strat,flips={f'{a}:{s}':v for (a,s),v in flips.items()},exact_family_histogram=hist,
                exact_rule_templates=templates,mean_walsh_logit_contrast={k:v/len(fs) for k,v in walsh.items()},
                mean_within_family_walsh_energy_share={k:v/len(fs) for k,v in energies.items()},
                fail_clustering={'name_pair':ccl(family_fails_by_name),'color_pair':ccl(family_fails_by_color)}),fam_success

def study(base,which):
    pins=PINS[which]; names=('panel/confirm_actor.json','panel/confirm_judge.json','actor-admissions/ledger.json','PREDICTION_SEAL.json','RESULTS.json')
    actor,judge,ledger,seals,official=(read(base/n,p) for n,p in zip(names,pins,strict=True))
    prompts=actor['prompts']; fs=[]
    for i,f in enumerate(judge['families']):
        key=tuple(map(tuple,f['key'])); need(f['rows']==list(range(i*8,i*8+8)),'family alignment')
        rows=[]
        for idx,truth in zip(f['rows'],f['answers'],strict=True):
            n,v,x,m,w,y,q=prompts[idx].split()
            need(v==w=='is' and set((n,m))==set(key[0]) and set((x,y))==set(key[1]) and q in (n,m),'fixture')
            mapping={n:x,m:y}
            c=(1 if mapping[key[0][0]]==key[1][0] else -1,1 if n==key[0][0] else -1,1 if q==key[0][0] else -1)
            need(mapping[q]==truth,'answer differs')
            rows.append((c,idx,truth))
        need(set(c for c,_,_ in rows)==set(product((1,-1),repeat=3)),'factorial')
        fs.append((key,rows))
    panelkeys={k for k,_ in fs}
    need(len(fs)==len(panelkeys)==120 and len(set(prompts))==960,'coverage')
    arms=('w1','w025') if which=='002' else ('clip1','clip2')
    expected={f'{a}-{s}-{i}' for a,s,i in product(arms,S,(128,1280))}
    need(expected==set(seals)==set(ledger)==set(official),'actor roster')
    results={}; successes={}
    for case,entry in sorted(seals.items()):
        c=ledger[case]; d=base/'actors'/entry['id']
        r=read(d/'run.json',entry['receipt_sha256'])
        need(verify_run(d)['valid'] and r['status']=='succeeded' and r['inputs']==c['inputs']
             and r['artifacts']['predictions.json']==entry['prediction_sha256'],'native actor verify')
        p=read(d/'output/predictions.json',entry['prediction_sha256'])
        need((p['actor_sha256'],p['checkpoint_sha256'],p['seed'],p['updates'],p['arm'],p['vocab_size'])
             ==(pins[0],c['checkpoint_sha256'],c['seed'],c['endpoint'],'gated32',287),'actor output identity')
        v=read(base/'actor-admissions'/case/'vocabulary.json',c['inputs']['vocabulary.json'])
        need(p['tokenizer_words']==v['words'] and len(p['logits'])==960,'vocab/shape')
        results[case],successes[case]=summarize(fs,p['logits'],{v:i+262 for i,v in enumerate(v['words'])})
        u=results[case]; o=official[case]['all']
        need(u['strata']['all']['exact']==o['exact_worlds'] and u['strata']['all']['relative_correct']==o['two_color_worlds']
             and u['exact_family_histogram'][8]==o['exact_families'],'official recount')
        if which=='003':
            need(all(u['flips'][a+':all']['both_exact']==o[k] for a,k in
              [('assignment','assignment_both_exact'),('order','clause_order_both_exact'),('query','query_both_exact')]),'pair recount')
    cross={}
    for a,b in combinations(sorted(x for x in successes if x.endswith('-1280')),2):
        pairs=Counter((str(int(a1))+str(int(b1))) for a1,b1 in zip(successes[a],successes[b],strict=True))
        af=pairs['00']+pairs['01']; bf=pairs['00']+pairs['10']
        cross[f'{a}/{b}']=dict(both_fail=pairs['00'],both_pass=pairs['11'],first_only_fails=pairs['01'],
            second_only_fails=pairs['10'],failure_expected_if_independent=af*bf/120,
            failure_jaccard=pairs['00']/(af+bf-pairs['00']) if af+bf else None)
    return dict(cases=results,crossed_family_failure_overlap=cross,actor_count=12,logit_row_count=11520), panelkeys

def main():
    top=Path(sys.argv[1])
    results={}; keys={}
    for w in ('002','003'): results[w],keys[w]=study(top/f'cog-role-{w}',w)
    need(not keys['002']&keys['003'],'held panels overlap')
    with Path(sys.argv[2]).open('x') as f:
        json.dump(dict(results=results,pinned_input_sha256=PINS,model_invocations=0,
                       addressability='Descriptive posthoc; not a fresh qualification test.'),f,sort_keys=True,indent=2,allow_nan=False)
        f.write('\n')
if __name__=='__main__':main()
