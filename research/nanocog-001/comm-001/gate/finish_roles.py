"""Report exact readout aggregates of frozen local ablation, with no inference."""
import json
from pathlib import Path
from statistics import mean
ARMS=('gated32','static32','attention32','gated48')
SEEDS=(1171,1271,1371)
def read(p):return json.loads(Path(p).read_bytes())
def main():
    data={**read('GATED32_RESULTS.json'),**read('COMPARATOR_RESULTS.json')}
    print('| Arm | Seed | Control 128 worlds | Control 128 all 8 | Control 1,280 worlds | Control 1,280 all 8 | FIT 1,280 all 8 | Pass |')
    print('|---|---:|---:|---:|---:|---:|---:|:---:|')
    for a in ARMS:
        for s in SEEDS:
            early,late=[data[f'{a}-{s}-{n}-control']['all'] for n in (128,1280)]
            fit=data[f'{a}-{s}-1280-fit']['all']
            passes=late['exact_worlds']>=912 and late['exact_families']>=108
            print(f"| {a} | {s} | {early['exact_worlds']}/960 | {early['exact_families']}/120 | {late['exact_worlds']}/960 | {late['exact_families']}/120 | {fit['exact_families']}/320 | {'yes' if passes else 'no'} |")
    print('\n## Training diagnostics (mean over last 32 updates)')
    print('| Arm | Seed | Prefix CE (nats) | Answer CE (nats) | Sibling foil loss | Preclip norm | Clipping rate |')
    print('|---|---:|---:|---:|---:|---:|---:|')
    for p in sorted(Path('.').glob('fits-*/*/output/steps.jsonl')):
        r=read(p.parents[1]/'run.json')
        rows=[json.loads(a) for a in p.read_text().splitlines()][-32:]
        names=['prefix_ce_nats_per_target','answer_ce_nats_per_target','paired_foil_softplus','raw_gradient_norm']
        vals=[mean(x[k] for x in rows) for k in names]
        vals+=[mean(x['raw_gradient_norm']>1 for x in rows)]
        print(f"| {r['name'].split()[1]} | {r['seed']} | "+" | ".join(f"{v:.4f}" for v in vals)+" |")
if __name__=='__main__':main()
