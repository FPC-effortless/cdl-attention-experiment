"""Generate semantic cases only; imports no model or historical dataset."""
import json
KEYS=tuple('r'+str(i) for i in range(4))
VALUES=tuple('v'+str(i) for i in range(8))
def generate():
    rows=[]
    for key in KEYS:
        for new in VALUES:
            for old in (None,*VALUES):
                events=[] if old is None else [dict(op='set',key=key,value=old)]
                events.append(dict(op='set',key=key,value=new))
                rows.append(dict(case_id=key+'-'+str(old)+'-'+new,key=key,old=old,new=new,
                    panel='baseline' if old is None else 'overwrite',
                    final_set_token=3*len(events)-1,
                    memory=dict(schema='woyengi.lifecycle.v1',keys=list(KEYS),events=events,composed_keys=[key])))
    return rows
def validate(rows):
    if len(rows)!=288:raise ValueError('case count')
    expected={(k,o,n) for k in KEYS for n in VALUES for o in (None,*VALUES)}
    actual=[(r['key'],r['old'],r['new']) for r in rows]
    if len(set(actual))!=len(actual) or set(actual)!=expected:raise ValueError('Cartesian coverage')
    for r in rows:
        events=([dict(op='set',key=r['key'],value=r['old'])] if r['old'] is not None else [])+[dict(op='set',key=r['key'],value=r['new'])]
        if r['memory']!=dict(schema='woyengi.lifecycle.v1',keys=list(KEYS),events=events,composed_keys=[r['key']]):raise ValueError('semantic drift')
        if r['panel']!=('baseline' if r['old'] is None else 'overwrite') or r['final_set_token']!=3*len(events)-1:raise ValueError('event boundary')
    return dict(cases=288,baselines=32,overwrites=256,diagonal=32,off_diagonal=224,planned_predictions=13824,model_calls=0)
if __name__=='__main__':
    rows=generate();validate(rows)
    print(json.dumps(rows,sort_keys=True,indent=2))

