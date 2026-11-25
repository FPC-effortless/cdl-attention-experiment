"""Prospectively construct model-free paired schedules from original TRAIN only."""
from collections import Counter,defaultdict
from hashlib import sha256
from itertools import combinations
from pathlib import Path
from random import Random
import json
R=Path(__file__).resolve().parent
TRAIN=R.parent/'cog-role-001/panels/train.json'
PIN='02607c19e760e164989865a54422eb7cd78fed7d117a7fa2ff6c4df9d700f9f8'
SEEDS=(1771,1871,1971)
STEPS=1280
ORDER=(0,4,1,5,2,6,3,7)
FOILS=((0,1),(2,3),(4,5),(6,7))
CONSIST=((0,2),(1,3),(4,6),(5,7))
STRATA=(('focal_first','distractor_second'),('focal_second','distractor_first'))
def need(ok,msg):
    if not ok:raise ValueError(msg)
def save(p,x):
    b=(json.dumps(x,sort_keys=True,allow_nan=False,separators=(',',':'))+'\n').encode()
    with p.open('xb') as f:f.write(b)
    return sha256(b).hexdigest()
def parse(prompt):
    a,w,x,b,v,y,q=prompt.split()
    need(w==v=='is' and a!=b and x!=y and q in (a,b),'invalid original role prompt')
    return (a,b),(x,y),q
def load():
    b=TRAIN.read_bytes();need(sha256(b).hexdigest()==PIN,'exact TRAIN digest')
    d=json.loads(b);need(set(d)=={'schema','role','families'} and d['role']=='TRAIN' and len(d['families'])==320,'role')
    world=d['families']; keys=[]; details={}; observed=set()
    for i,row in enumerate(world):
        need(len(row['prompts'])==len(row['answers'])==8,'family shape')
        for j,(prompt,y) in enumerate(zip(row['prompts'],row['answers'],strict=True)):
            names,colors,q=parse(prompt); answer=dict(zip(names,colors))[q]
            k=tuple(sorted(names)),tuple(sorted(colors))
            if j==0:keys.append(k)
            need(k==keys[-1] and y==answer and prompt not in observed,'true target or cell reuse')
            observed.add(prompt)
            details[i,j]=(names,colors,q)
    need(len(set(keys))==320 and len(observed)==2560,'TRAIN exact support')
    return world,keys,details
def edge_graph(keys):
    b=defaultdict(list)
    for i,(names,colors) in enumerate(keys):
        for n in names:b[n,colors].append(i)
    edges=[(i,j,n,c) for (n,c),v in sorted(b.items()) for i,j in combinations(v,2)]
    v=defaultdict(list)
    for e,(i,j,n,c) in enumerate(edges):
        v[i].append(e);v[j].append(e)
    need(len(edges)==444 and len(v)==297 and sum(len(y)>1 for y in b.values())==186,'expected exact edges')
    return edges,v
def row_refs(edge,indices,detail,strata):
    i,j,anchor,colors=edge
    # Mirror exact prompt semantics, not the order of family-key names.
    out=[]
    for s in strata:
        for family in indices:
            focal=next(n for n in detail[family,0][0] if n!=anchor)
            for ai in (0,1):
                assignment=colors if ai==0 else colors[::-1]
                found=[]
                for r in range(8):
                    names,vals,q=detail[family,r]
                    focal_first=names[0]==focal
                    slot = (('focal' if q==focal else 'distractor') + '_' +
                            (('first' if focal_first else 'second') if q==focal else ('second' if focal_first else 'first')))
                    if slot==s and dict(zip(names,vals))[focal]==assignment[0]:
                        found.append(r)
                need(len(found)==1,'not one approved row per family/assignment/stratum')
                out.append([family,found[0]])
    return out
def build(seed,world,keys,detail,edges,graph):
    rng=Random(seed ^ 0xC060001)
    slots={i:[(i,k) for k in range(4)] for i in graph}
    picks={}
    # Deterministic incidence matching: reserve each exact available edge
    # once into one of its endpoints' four primary-family epoch slots.
    search=[[*slots[i],*slots[j]] for i,j,_,_ in edges]
    for row in search:rng.shuffle(row)
    queue=list(range(len(edges)));rng.shuffle(queue)
    def assign(e,seen):
        for s in search[e]:
            if s in seen:continue
            seen.add(s)
            if s not in picks or assign(picks[s],seen):
                picks[s]=e
                return True
        return False
    need(all(assign(e,set()) for e in queue),'cannot reserve full edge coverage')
    load=Counter(picks.values())
    fill=[(i,k) for i in sorted(graph) for k in range(4) if (i,k) not in picks];rng.shuffle(fill)
    for i,k in fill:
        v=graph[i].copy();rng.shuffle(v)
        a=min(v,key=load.__getitem__);picks[i,k]=a;load[a]+=1
    # Anchor identity can switch across epoch slots. Guarantee two visits
    # in each *actual primary canonical rendering* as well as best feasible
    # total focal-first/focal-second balance across all covered visits.
    # DP chooses only from six ways to give each primary 2+2 renders, so the
    # chosen anchor cannot erase any of its 8 original/unedited cells.
    cover=sorted(graph); states={0:()}
    for i in cover:
        roles=[next(n for n in keys[i][0] if n!=edges[picks[i,e]][2])==keys[i][0][0] for e in range(4)]
        options=list(combinations(range(4),2))
        rng.shuffle(options)
        options.sort(key=lambda v:-sum((e in v)!=((e+1) in v) for e in range(3)))
        next_states={}
        for v in options:
            matches=sum(roles[e]==(e in v) for e in range(4))
            for k,prev in states.items():
                next_states.setdefault(k+matches,prev+(v,))
        states=next_states
    target=min(states,key=lambda n:(abs(n-594),n))
    need(abs(target-594)<=1,'registered global geometry imbalance too high')
    rendered=dict(zip(cover,states[target],strict=True))
    primary_rng=Random(seed^0xC060001)
    steps=[];seen=Counter(); primary=Counter(); strata=Counter(); foil_count=0; pair_count=0
    for ep in range(4):
        order=list(range(len(world)));primary_rng.shuffle(order)
        for i in order:
            e=picks.get((i,ep))
            if e is None:
                batch={'kind':'ordinary','primary':i,'rows':[[i,j] for j in ORDER]}
            else:
                v=edges[e]; partner=v[0] if v[1]==i else v[1]
                primary_focal=next(n for n in keys[i][0] if n!=v[2])
                focal_first=(primary_focal==keys[i][0][0])==(ep in rendered[i])
                ss=STRATA[0 if focal_first else 1]
                rows=row_refs(v,(i,partner),detail,ss)
                batch={'kind':'paired','primary':i,'edge':e,'strata':ss,'rows':rows}
                strata.update(ss);pair_count+=4
                for n,m in CONSIST:
                    ai,aj=rows[n],rows[m]
                    x,y=(world[u]['prompts'][r].split() for u,r in (ai,aj))
                    dx=[(a,b) for a,b in zip(x,y,strict=True) if a!=b]
                    first=ss[n//4]
                    need(len(dx)==(2 if first.startswith('focal') else 1) and len(set(dx))==1
                      and world[ai[0]]['answers'][ai[1]]==world[aj[0]]['answers'][aj[1]],'matching raw edits/answer')
            rr=batch['rows'];need(len(set(tuple(x) for x in rr))==8,'duplicate world per update')
            for z1,z2 in FOILS:
                u,r=rr[z1];v,t=rr[z2]
                s1,c1,q1=detail[u,r];s2,c2,q2=detail[v,t]
                need(u==v and s1==s2 and q1==q2 and c1==c2[::-1],'genuine original assignment foil')
            foil_count+=4
            seen.update(tuple(x) for x in rr)
            primary.update(tuple(x) for x in rr if x[0]==i)
            steps.append(batch)
    degree=Counter(i for e,m in load.items() for i in edges[e][:2] for _ in range(m))
    per_name=Counter(n for (family,row),count in seen.items() for n in (detail[family,row][2],) for _ in range(count))
    need(len(steps)==STEPS and len(seen)==len(primary)==2560 and sum(seen.values())==10240
         and len(load)==444 and pair_count==4*1188 and foil_count==5120 and sum(strata.values())==2376 and all(abs(v-594)<=1 for v in strata.values()),
         str((len(steps),len(seen),len(primary),sum(seen.values()),len(load),pair_count,foil_count,strata)))
    return dict(schema='cog-role-name-consist.schedule.v1',seed=seed,train_sha256=PIN,steps=steps),dict(
         paired_updates=1188,ordinary_updates=92,updates=1280,worlds=10240,dense_input_tokens=71680,
         paired_foil_pairs=5120,paired_consistency_pairs=4752,distinct_worlds=2560,
         distinct_primary_worlds=len(primary),unique_edges=len(load),stratum_batch_counts=strata,
         edge_use_histogram=Counter(load.values()),world_use_histogram=Counter(seen.values()),
         primary_world_use_histogram=Counter(primary.values()),paired_family_presence_histogram=Counter(degree.values()),
         per_queried_literal_exposure=per_name)
def main():
    world,keys,details=load(); es,graph=edge_graph(keys)
    out=R/'train_schedule';out.mkdir()
    r={}
    for seed in SEEDS:
        schedule,t=build(seed,world,keys,details,es,graph)
        r[str(seed)]={'sha256':save(out/f'schedule-{seed}.json',schedule),'summary':t}
    manifest=dict(schema='cog-role-name-consist.audit.v1',train_sha256=PIN,covered_families=len(graph),candidate_pairs=len(es),
        all_focal_names=sorted(set(n for i,j,anchor,_ in es for k in (i,j) for n in keys[k][0] if n!=anchor)),
        same_schedule_both_arms=True,seeds=r,uncovered=sorted(set(range(len(world)))-set(graph)),
        foil_index_pairs=FOILS,consistency_index_pairs=CONSIST)
    print(save(out/'manifest.json',manifest))
if __name__=='__main__':main()
