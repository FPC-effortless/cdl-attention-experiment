"""Strict 30-fit screen aggregation. No official validation/test access."""
import argparse,json,math,statistics
from pathlib import Path
from optimizers import RECIPES,RATES
SEEDS=(9101,9109)
def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    rows=[json.loads(p.read_text()) for p in a.input.rglob('result.json')]
    expected={(r,i,s) for r in RECIPES for i in range(3) for s in SEEDS};found={}
    for r in rows:
        c=r['config'];k=(c['recipe'],c['rate_index'],c['seed']);assert k not in found and k in expected and c['rate']==RATES[k[0]][k[1]];assert r['terminal'] and r['reload_verified'] and r['targets']==12582912 and math.isfinite(r['bpb']);found[k]=r
    assert set(found)==expected,(len(found),expected-set(found))
    assert len({r['config']['data_manifest'] for r in rows})==1 and len({r['eval_targets'] for r in rows})==1
    for seed in SEEDS:assert len({r['config']['data_order'] for r in rows if r['config']['seed']==seed})==1
    table=[]
    for recipe in RECIPES:
        cells=[]
        for i in range(3):
            rs=[found[recipe,i,s] for s in SEEDS]
            cells.append(dict(rate=RATES[recipe][i],rate_index=i,seed_bpb={str(s):found[recipe,i,s]['bpb'] for s in SEEDS},mean_bpb=statistics.mean(r['bpb'] for r in rs),mean_train_seconds=statistics.mean(r['train_seconds'] for r in rs),optimizer_state_bytes=statistics.mean(r['optimizer_state_bytes'] for r in rs),mean_peak_rss_kib=statistics.mean(r['peak_rss_kib'] for r in rs)))
        best=min(cells,key=lambda r:r['mean_bpb']);table.append(dict(recipe=recipe,screen_best=best,edge_of_rate_grid=best['rate_index'] in (0,2),all_rates=cells))
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(dict(status='DEVELOPMENT_SCREEN_ONLY',fits=30,results=sorted(table,key=lambda x:x['screen_best']['mean_bpb']),note='Two seeds; tuned on reused development set; no full-budget or mechanism qualification. No winner promoted.'),indent=2));print(a.out.read_text())
if __name__=='__main__':main()
