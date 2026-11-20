"""Independent arithmetic on published predictions and closed-loop traces."""
import json
import math
from pathlib import Path
from statistics import median
HERE=Path(__file__).resolve().parent
R=HERE/'results'
def load(path): return json.loads(path.read_text())
def ap(rows):
    ranks={}
    for r in rows:
        rank=r[5]; n,ok=ranks.get(rank,(0,0));ranks[rank]=n+1,ok+r[4]
    p=n=weighted=0
    for score in sorted(ranks,reverse=True):
        k,j=ranks[score];p+=j;n+=k;weighted+=j*p/n
    return weighted/p if p else None
def run():
    checks=0; physical=0; ranking=0
    for seed in (67003,67009,67021):
        scores=load(R/str(seed)/'issue-results.json')
        rows=load(R/str(seed)/'issue-predictions.json')
        for head,part in scores.items():
            for split in ('test','external','validation'):
                data=[r for r in rows if r[0]==head and r[1]==split]
                assert len({r[3] for r in data})==len(data)
                stats=part[split]
                assert abs(stats['ap']-ap(data))<1e-10
                assert sum(r[4] and r[6] for r in data)==stats['tp']
                assert sum(not r[4] and r[6] for r in data)==stats['fp']
                ranking+=1
        for arm in ('candidate','control'):
            control_seed={67003:5101,67009:5107,67021:5113}[seed]
            path=R/str(control_seed)/('acquisition-'+arm)/'result.json'
            data=load(path)
            for row in data['session']:
                assert row['exact']==(row['route'] and not row['repaired'] and row['final']==row['expected'])
                physical+=1
            for scale in data['scaling']:
                assert len(scale['rows'])==256 and all(r[-1] for r in scale['rows'])
                checks+=len(scale['rows'])
    confirms=load(R/'confirmation.json')
    for split,stat in confirms['summary'].items():
        part=[r for r in confirms['rows'] if r[0]==split]
        assert stat['tp']==sum(y and a for _,_,_,y,p,a,w in part)
        assert stat['fp']==sum(not y and a for _,_,_,y,p,a,w in part)
    return dict(ranking_tables=ranking,physical_traces=physical,scaling_answers=checks)
if __name__=='__main__': print(json.dumps(run(),sort_keys=True))
