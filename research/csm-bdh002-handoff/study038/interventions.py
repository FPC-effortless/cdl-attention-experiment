"""Post-hoc reference interventions over the original episodes."""
import json
import pandas as pd
from experiment import ROOT, ALIAS, approval_reference

raw = pd.read_csv(ROOT/'raw.csv.gz')
oracle = raw[raw.arm.eq('oracle')].set_index(['seed','episode','frame','reforder','approval'])
out = []
for row in raw[(raw.arm.eq('learned')) & raw.frame.isin(['explicit_groups','unseen_alias'])].itertuples():
    weights = json.loads((ROOT/f'weights-{row.seed}.json').read_text())
    if row.frame == 'explicit_groups':
        prefix, clauses = row.text.split(' compare ')
        governed = approval_reference(prefix + ' ' + clauses.split(' the ',2)[2])
        weights_used = weights
    else:
        governed = approval_reference(row.text)
        # Supply two labeled word-to-position associations.
        # Use position 0 for earlier and 1 for later.
        weights_used = dict(zip(ALIAS,(-1.,1.)))
    score = weights_used.get(governed,0.)
    pick = int(score>0) if score else -1
    gold = oracle.loc[(row.seed,row.episode,row.frame,row.reforder,row.approval)]
    assert pick == gold['pick']
    out.append((row.seed,row.episode,row.frame,row.reforder,row.approval,row.reference,governed,pick,int(gold['pick'])))
p = pd.DataFrame(out,columns=['seed','episode','frame','reforder','approval','original_reference','reference','pick','gold'])
p.to_csv(ROOT/'interventions.csv',index=False)
print(p.groupby(['seed','frame']).size())
