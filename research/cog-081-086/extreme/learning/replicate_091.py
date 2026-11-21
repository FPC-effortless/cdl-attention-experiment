"""Fresh, post-initial 091 replication: frozen policy, new disjoint maze seeds.

No held-out result is used for feature construction or alpha selection. This is
an added replication diagnostic, not the original preregistered reserved panel.
"""
from __future__ import annotations
import hashlib,json,random,sys
from pathlib import Path
from time import perf_counter_ns
from collections import Counter
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(HERE.parent))
from difficulty import make_maze_challenge,maze_changed,encode_puzzle,digest
from score_ladder import environment
from maze_transfer_091 import collect_training,calibrate,explore,preregistered_budget

SEEDS=(9803,9809,9811,9817)

def main():
    cases=[json.loads(x) for x in (HERE.parent/'results'/'cases.jsonl').read_text().splitlines()]
    traces,model,options,train_ns=collect_training(cases)
    alpha,calibration=calibrate(cases,model)
    prior={r['input_sha256'] for r in cases if r['domain']=='maze'}
    rows=[];extra=[]
    for seed in SEEDS:
        for level in range(1,11):
            m,ref=make_maze_challenge(seed*100003+level*131,level)
            fingerprint=digest(('maze',encode_puzzle(m)))
            assert fingerprint not in prior
            prior.add(fingerprint)
            shifted=None
            if level>=8:
                shifted,_=maze_changed(m,ref.path)
                assert shifted is not None
            for world,maze in [('original',m)]+([('shift',shifted)] if shifted else []):
                options_in_mode=['heuristic','learned','shuffled','random']
                rng=random.Random(seed*100+level)
                rng.shuffle(options_in_mode)
                r={}
                for arm in options_in_mode:
                    x=explore(maze,mode=arm,model=model,alpha=alpha,
                              seed=seed*131+level*17+(world=='shift'),
                              budget=preregistered_budget(maze.size))
                    r[arm]={k:v for k,v in x.items() if k not in ('walk','observations','verified_path')}
                rows.append(dict(seed=seed,level=level,world=world,size=m.size,
                                 input_sha256=fingerprint,arms=r))
    totals={a:dict(total=len(rows),solved=sum(x['arms'][a]['solved'] for x in rows),
                   actions=sum(x['arms'][a]['actions'] for x in rows),
                   wall_ms=round(sum(x['arms'][a]['elapsed_ns'] for x in rows)/1e6,3))
            for a in ('heuristic','learned','shuffled','random')}
    out=dict(study='COG-ATTN-091 additional fresh-seed replication of local maze skill controller',
             training_worlds=len(traces),frozen_alpha=alpha,seeds=list(SEEDS),total_worlds=len(rows),
             skill_features=len(model),acquired_cog066_options=len(options),
             no_reference_path_test_time=True,controls=totals,rows=rows,
             caveats=['post-initial fresh seed cohort, not independent external researcher',
                      'maze generator family is the same as training and initial reserved',
                      'controller is a lightweight novel local learner, not an end-to-end COG-ATTN model'])
    target=HERE/'results'/'maze-091-fresh.json';target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps(totals,indent=2))
    return out

if __name__=='__main__':main()
