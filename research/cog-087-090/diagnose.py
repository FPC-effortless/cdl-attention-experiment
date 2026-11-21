"""Score the two locked heads on held rollout states; oracle is evaluator-only."""
from dataclasses import replace
import json
from statistics import mean
from panel import ROOT,inputs
from search import solve
from skills import suffix,features,predict
def run():
    model=json.loads((ROOT/'results/model.json').read_bytes())
    rows=[]
    for part in ('held','rep'):
        for t in inputs(part,model):
            actions=solve(t)['actions'];s=t.start
            at={0,len(actions)//2,len(actions)-1}
            for j,a in enumerate(actions):
                if j in at:
                    u=replace(t,start=s)
                    seq=() if t.kind=='maze' else suffix(t,s,model['book'])
                    candidate=bool(seq) if t.kind!='maze' else bool(model['corridor_support'])
                    f=features(t,s,t.neighbors(s),seq)
                    gain=predict(model['tree'],f);app=predict(model['app_tree'],f)
                    base=solve(u);forced=solve(u,'structural',model=model,first=True)
                    assert base['steps']==forced['steps']
                    rows.append(dict(task=t.id,index=j,candidate=candidate,applicability=app,
                        realized_applicable=forced['certified']>0,predicted_savings=gain,
                        realized_savings=base['attempts']-forced['attempts'],
                        realized_charged_savings=base['charged']-forced['charged'],
                        activate=candidate and app>=.5 and gain>model['tau'][t.kind]))
                s=dict(t.neighbors(s))[a]
    return rows
if __name__=='__main__':
    (ROOT/'results/paired-state-diagnostic.json').write_text(json.dumps(run(),separators=(',',':'))+'\n')
