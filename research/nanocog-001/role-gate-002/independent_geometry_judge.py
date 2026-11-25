"""Geometry-only seal and independent feasibility replay. No torch or model import."""
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import permutations
from math import exp, fsum, hypot, isclose, isfinite
from pathlib import Path
import json,sys
from trainlab.runner import verify_run

H=Path('geometry_admissions_v2')
R=Path('geometry_runs_v2')
LEDGER='5f998d9680c416e1b192667b55864b8039593b1dd275d3989aced408eb6c933e'
REQUIRED={'geometry-1171','geometry-1371'}
def need(t,msg):
    if not t:raise ValueError(msg)
def sha(p):return sha256(p.read_bytes()).hexdigest()
def read(p,pin=None):
    b=p.read_bytes()
    need(pin is None or sha256(b).hexdigest()==pin,f'digest {p}')
    return json.loads(b)
def sealed_dump(p,o):
    with p.open('x') as f:json.dump(o,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    return sha(p)

def seal(out):
    ledger=read(H/'ledger.json',LEDGER); need(set(ledger)==REQUIRED,'recipe roster')
    proofs={}
    for tag,c in ledger.items():
        d=H/tag
        r=read(d/'cli-output.json'); run=R/r['id']
        need(r==read(run/'run.json') and verify_run(run)['valid'],'native integrity')
        need(r['status']=='succeeded' and r['exit_code']==0 and 0<r['elapsed_seconds']<=45
          and r['execution_purpose']=='engineering_diagnostic' and r['inputs']==c['inputs']
          and r['name']==tag and r['seed']==c['seed']
          and r['metrics']=={'probability_rows':256,'model_forwards':32,'intervention_forwards':0,'optimizer_updates':0}
          and set(r['artifacts'])=={'geometry.json'},'native scope')
        need(all(sha(d/k)==v for k,v in c['inputs'].items()) and sha(d/'recipe.json')==c['recipe_sha256']
          and read(d/'recipe.json')==read(run/'experiment.json'),'exact runner closure')
        proofs[tag]={'run_id':r['id'],'run_sha256':sha(run/'run.json'),'geometry_sha256':r['artifacts']['geometry.json']}
    need({i['run_id'] for i in proofs.values()}=={d.name for d in R.iterdir() if d.is_dir()},'extra/unrecorded run')
    print(sealed_dump(out,dict(ledger_sha256=LEDGER,receipts=proofs)))

def independent_radius(p,q):
    changes=[y-x for x,y in zip(p,q,strict=True)]
    bias=fsum(changes)*.25; changes=[x-bias for x in changes]
    n=hypot(*changes)
    if n<=1e-12:return (0.,'tiny_native_donor')
    if min(p)==0.:return (0.,'native_support_zero')
    toward=[a/n for a in changes]
    def feasible(d):
        return min(-x/y for x,y in zip(p,d,strict=True) if y<0)
    fixed=feasible(toward)
    candidates=[]
    for sham in permutations(toward):
        if abs(fsum(a*b for a,b in zip(toward,sham,strict=True)))<=.5:
            candidates.append(min(n,.99*min(fixed,feasible(sham))))
    if len(candidates)==0:return (0.,'no_angular_sham')
    r=max(candidates)
    return (r,'active') if r>=1e-9 else (0.,'tiny_common_feasible')

def verified_rows(pkt):
    source=pkt['rows'];need(len(source)==256,'probability roster')
    normalized=[]
    for i,row in enumerate(source):
        p=row['p_fp32']; s=row['scores_fp32']
        need(row['row']==i and row['donor']==i^1 and len(p)==len(s)==4
             and all(0<=x<=1 for x in p) and all(isfinite(x) for x in s) and abs(fsum(p)-1)<=2e-7,'row/numeric')
        shift=[exp(x-max(s)) for x in s]; z=fsum(shift)
        need(all(isclose(x,y/z,abs_tol=1e-7,rel_tol=1e-6) for x,y in zip(p,shift,strict=True)),'scores/probability disagreement')
        normalized.append([x/fsum(p) for x in p])
    radii=[];statuses=Counter()
    for i,row in enumerate(source):
        a,st=independent_radius(normalized[i],normalized[i^1]);w=row['intended']
        need(w['status']==st and isclose(a,w['radius'],abs_tol=1e-14,rel_tol=1e-12),'independent geometry mismatch')
        if st=='active':
            p=normalized[i]
            for branch,actual in [('toward','toward_actual'),('sham','sham_actual')]:
                u=w[branch]
                need(len(u)==4 and min(u)>0 and abs(fsum(u)-1)<=2e-14
                     and isclose(hypot(*(x-y for x,y in zip(u,p,strict=True))),a,abs_tol=4e-16,rel_tol=3e-7)
                     and isclose(w[actual],a,abs_tol=4e-16,rel_tol=3e-7),'unbalanced intended geometry')
        statuses[st]+=1; radii.append(a)
    return radii,statuses

def score(seal,pin,out):
    z=read(seal,pin);ledger=read(H/'ledger.json',LEDGER)
    need(z['ledger_sha256']==LEDGER and set(z['receipts'])==set(ledger)==REQUIRED,'sealed roster')
    branches={}
    for tag,proof in z['receipts'].items():
        base=H/tag;c=ledger[tag];d=R/proof['run_id'];r=read(d/'run.json',proof['run_sha256'])
        need(r==read(base/'cli-output.json') and verify_run(d)['valid'] and r['artifacts']['geometry.json']==proof['geometry_sha256'],'seal mismatch')
        pkt=read(d/'output/geometry.json',proof['geometry_sha256'])
        need((pkt['schema'],pkt['seed'],pkt['actor_sha256'],pkt['checkpoint_sha256'])==
          ('cog-role-gate-002.geometry.v1',c['seed'],c['actor_sha256'],c['checkpoint_sha256']),'geometry binding')
        actor=read(base/'actor.json',c['actor_sha256'])['prompts'];need(len(actor)==len(set(actor))==256,'actor geometry')
        # Use histories, not answers. Verify four disjoint histories in each
        # family and both opposite-query members per history.
        for j in range(0,256,8):
            h=defaultdict(list)
            for n,s in enumerate(actor[j:j+8],j):
                w=s.split()
                need(len(w)==7 and w[1]==w[4]=='is' and w[-1] in (w[0],w[3]),'prompt geometry')
                h[tuple(w[:6])].append(n)
            need(len(h)==4 and all(len(ii)==2 and ii[0]^1==ii[1] and actor[ii[0]].split()[-1]!=actor[ii[1]].split()[-1] for ii in h.values()),'opposite query geometry')
        rr,ss=verified_rows(pkt)
        yes=[rr[i]>=.1 and rr[i+1]>=.1 for i in range(0,256,2)]
        counts=[sum(yes[i:i+4]) for i in range(0,128,4)]
        row_support=sum(x>=.1 for x in rr)
        need(row_support==pkt['intended_radius_at_least_point1'],'native row count differs')
        hist=sum(yes); fam=sum(x>=3 for x in counts)
        branches[tag]=dict(geometry_run=proof['run_id'],geometry_sha256=proof['geometry_sha256'],
            qualifying_histories=hist,total_histories=128,qualifying_families=fam,total_families=32,
            ideal_radius_at_least_point1_rows=row_support,total_rows=256,history_count_by_family_histogram=Counter(counts),
            geometry_statuses=ss,gate_met=hist>=103 and fam>=26,descriptive_geometry_summary=pkt['summary'],
            natural_model_forwards=r['metrics']['model_forwards'],intervention_forwards=0,optimizer_updates=0)
    ok=all(b['gate_met'] for b in branches.values())
    record=dict(seal_sha256=pin,qualified_for_review_of_later_directional_worker=ok,
         action='GEOMETRY_PREREQUISITE_ONLY' if ok else 'HOLD_INTERVENTIONS_INSUFFICIENT_USEFUL_DOSE',
         actual_dtype_interventions_tested=False,scientific_model_promoted=False,branches=branches)
    print(sealed_dump(out,record));print(json.dumps(record,sort_keys=True,allow_nan=False))

if __name__=='__main__':
    if sys.argv[1]=='seal':seal(Path(sys.argv[2]))
    elif sys.argv[1]=='score':score(Path(sys.argv[2]),sys.argv[3],Path(sys.argv[4]))
    else:raise ValueError('expected seal or score')
