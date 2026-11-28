"""Fail-closed matched development comparison; no winner from oracle accuracy."""
import argparse,hashlib,json,math,statistics
from pathlib import Path
from experiment import ARMS,SEEDS,STEPS,BATCH
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();rows={}
    for p in a.input.rglob('result.json'):
        r=json.loads(p.read_text());c=r['config'];k=(c['arm'],c['seed']);assert k not in rows and k[0] in ARMS and k[1] in SEEDS
        assert r['status']=='TERMINAL_DEVELOPMENT_ONLY' and r['reload_verified'] and c['steps']==STEPS and c['examples']==STEPS*BATCH
        assert sha(p.parent/'checkpoint.pt')==r['checkpoint_sha256'] and sha(p.parent/'predictions.jsonl')==r['predictions_sha256']
        for n,v in r['panels'].items():
            assert v['n']==(128 if n.startswith('long_') else 512) and 0<=v['learned']<=v['n'] and 0<=v['paired_both']<=v['paired_n']
        rows[k]=r
    assert set(rows)=={(k,s) for k in ARMS for s in SEEDS},('incomplete',len(rows))
    assert len({json.dumps(r['config']['source'],sort_keys=True) for r in rows.values()})==1
    for s in SEEDS:
        assert len({json.dumps(rows[k,s]['config']['split_sha256'],sort_keys=True) for k in ARMS})==1
        assert len({rows[k,s]['config']['train_order_sha256'] for k,v in ARMS.items() if v[3] in ('random','mixed')})==1
    out=[]
    for arm in ARMS:
        q=[rows[arm,s] for s in SEEDS]
        def per(panel,field,den='n'):
            vals=[r['panels'][panel] for r in q];return [v[field]/v[den] if v[den] else None for v in vals]
        panel={}
        for p in q[0]['panels']:
            panel[p]={k:per(p,k,d) for k,d in [('learned','n'),('oracle','n'),('removed','n'),('address_correct','address_n'),('paired_both','paired_n'),('revision_both','paired_n'),('revision_non_target_exact','revision_non_target_n')]}
        out.append(dict(arm=arm,recipe=ARMS[arm],params=q[0]['config']['parameters'],trainable=q[0]['config']['trainable_parameters'],state_bytes=q[0]['panels']['two_names']['state_bytes'],training_seconds=[r['training_seconds'] for r in q],panels=panel,eligible_for_confirmation=all(v>=.9 for v in panel['held_syntax']['learned']) and all(v is not None and v>=.9 for v in panel['held_syntax']['paired_both'])))
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(dict(status='DEVELOPMENT_ONLY',terminal_fits=len(rows),original_014_replay_credit=0,original_cognition001_credit=0,selection='Pareto comparison only; learned access/paired revisions, state cost and baseline competence. Oracle not a selection score. New sealed confirmation and real-language transfer required.',arms=out),indent=2)+'\n');print(a.out.read_text())
if __name__=='__main__':main()
