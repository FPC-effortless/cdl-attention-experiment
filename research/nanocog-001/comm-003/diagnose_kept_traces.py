"""Read-only decomposition: answer CE = sibling choice + -log(pair mass)."""
import json
from pathlib import Path
from statistics import mean
from trainlab.runner import verify_run

here=Path(__file__).resolve().parent
roots={'clip1':here.parent/'cog-role-002'/'fits','clip2':here/'fits'}
out={}
for arm,root in roots.items():
    for d in sorted(root.iterdir()):
        if not d.is_dir() or not (d/'run.json').is_file(): continue
        if not verify_run(d)['valid']: raise ValueError(d)
        r=json.loads((d/'run.json').read_text())
        rows=[json.loads(line) for line in (d/'output/steps.jsonl').read_text().splitlines()]
        if len(rows)!=1280: raise ValueError(d)
        for ep in (128,1280):
            win=rows[ep-32:ep]
            row={
              'raw_prefix_ce':mean(s['prefix_ce_nats_per_target'] for s in win),
              'raw_answer_ce':mean(s['answer_ce_nats_per_target'] for s in win),
              'sibling_choice_nats':mean(s['paired_foil_softplus'] for s in win),
              'pair_mass_nats':mean(s['answer_ce_nats_per_target']-s['paired_foil_softplus'] for s in win),
              'raw_gradient_norm_mean':mean(s['raw_gradient_norm'] for s in win),
              'raw_gradient_above_1':sum(s['raw_gradient_norm']>1 for s in win),
              'raw_gradient_above_2':sum(s['raw_gradient_norm']>2 for s in win),
            }
            out[f"{arm}-{r['seed']}-{ep}"]=row
if len(out)!=12:raise ValueError('missing role trajectories')
print(json.dumps(out,indent=2,sort_keys=True))
