"""Require all terminal fits and derive a multiobjective Pareto frontier."""
import argparse,hashlib,json,statistics,math
from pathlib import Path
from integrated import ARMS
from train import SEEDS,STEPS,BATCH,T,KB,CORPUS

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();rows={}
    for p in a.input.rglob('result.json'):
        r=json.loads(p.read_text());c=r['config'];k=(c['arm'],c['seed']);assert k not in rows and k[0] in ARMS and k[1] in SEEDS and tuple(c['recipe'])==ARMS[k[0]]
        assert r['status']=='TERMINAL_DEVELOPMENT_ONLY' and r['terminal'] and r['reload_verified'] and c['steps']==STEPS and c['targets']==STEPS*BATCH*T and c['examples4']==c['examples8']==STEPS*KB and c['corpus']==CORPUS
        for file,key in [('checkpoint.pt','checkpoint_sha256'),('model.safetensors','model_sha256'),('metrics.json','metrics_sha256')]:assert sha(p.parent/file)==r[key]
        q=json.loads((p.parent/'receipt.json').read_text());assert q['terminal'] and q['step']==STEPS and q['sha256']==r['checkpoint_sha256'] and q['config']==c
        m=json.loads((p.parent/'metrics.json').read_text());assert m['text']['targets']==5029184 and m['text']['bpb']==r['language_bpb'] and m['cognition4']==r['cognition4'] and m['cognition8']==r['cognition8'];assert math.isfinite(r['language_bpb'])
        for x in r['cognition4'].values():assert x['n']==256 and x['pairs']>0
        for x in r['cognition8'].values():assert x['n']==256 and x['pairs']>0
        rows[k]=r
    assert set(rows)=={(a,s) for a in ARMS for s in SEEDS},('incomplete',len(rows));assert len({json.dumps(r['config']['source'],sort_keys=True) for r in rows.values()})==1
    for s in SEEDS:
        assert len({rows[a,s]['config']['text_order_sha256'] for a in ARMS})==1 and len({json.dumps(rows[a,s]['config']['knowledge_splits'],sort_keys=True) for a in ARMS})==1
        assert len({rows[a,s]['knowledge_draw_sha256'] for a in ARMS if ARMS[a][3]=='random_full'})==1
    table=[]
    for arm,recipe in ARMS.items():
        q=[rows[arm,s] for s in SEEDS];v={'arm':arm,'recipe':recipe,'parameters':q[0]['config']['parameters'],'state_bytes_13':q[0]['state_bytes_13'],'training_seconds':[x['training_seconds'] for x in q]}
        metrics={'language_bpb':[x['language_bpb'] for x in q],'four_learned':[x['cognition4']['full']['correct']/256 for x in q],'four_paired':[x['cognition4']['full']['both_correct']/x['cognition4']['full']['pairs'] for x in q],'eight_held':[x['cognition8']['held_full']['correct']/256 for x in q],'eight_paired':[x['cognition8']['held_full']['both_correct']/x['cognition8']['held_full']['pairs'] for x in q],'eight_long':[x['cognition8']['held_long']['correct']/256 for x in q],'revision':[x['cognition8']['revision']['both_correct']/x['cognition8']['revision']['pairs'] for x in q],'reset':[x['cognition8']['revision']['reset_correct']/256 for x in q]}
        v['metrics']={k:dict(mean=statistics.mean(x),seedwise=x) for k,x in metrics.items()};v['eligible_for_confirmation']=all(x>=.9 for k in ('four_paired','eight_paired','revision') for x in metrics[k]);table.append(v)
    def vector(x):
        z=x['metrics'];return (z['language_bpb']['mean'],*[-z[k]['mean'] for k in ('four_paired','eight_paired','eight_long','revision','reset')],x['state_bytes_13'])
    def dominates(x,y):return all(a<=b for a,b in zip(vector(x),vector(y))) and any(a<b for a,b in zip(vector(x),vector(y)))
    primary=[x for x in table if x['recipe'][3]=='random_full'];front=[x['arm'] for x in primary if not any(dominates(y,x) for y in primary if y is not x)]
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(dict(status='DEVELOPMENT_ONLY',terminal_fits=len(rows),pareto_primary=front,results=table,note='No weighted score or automatic architecture promotion. Controls with oracle training or fixed worlds separate. Timing heterogeneous; inference cost qualification later. Sealed confirmation worlds and official WikiText validation/test not evaluated.'),indent=2));print(a.out.read_text())
if __name__=='__main__':main()
