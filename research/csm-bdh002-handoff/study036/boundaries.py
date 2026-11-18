"""Post-hoc query diagnostic: provide the two candidate spans but hide their polarity."""
import pandas as pd
from study import ROOT, D, feature, b, np

raw=pd.read_csv(ROOT/'raw.csv.gz',dtype={'perm':str},keep_default_na=False)
out=[]
for (seed,mode,layout,episode), rows in raw[(raw.perm=='012') & (raw['mode']!='oracle')].groupby(['seed','mode','layout','episode']):
    keys=rows.gold_key.tolist()
    assert len(set(keys))==2
    ck=np.load(ROOT/f"fit-{seed}-{int(mode=='wide')}.npz")
    weights=dict(zip(ck['names'],ck['coef'][0]))
    pair=[]
    for row in rows.itertuples():
        words=row.text.split()
        scores=[]
        for k in keys:
            z=feature(words,*b.find(row.text,k),mode=='wide')
            scores.append(sum(weights.get(n,0.)*v for n,v in z.items()))
        j=int(b.choose(np.asarray(scores)))
        pair.append(j>=0 and keys[j]==row.gold_key)
    out.append([seed,mode,layout,episode,all(pair)])
result=pd.DataFrame(out,columns=['seed','mode','layout','episode','pair'])
result.to_csv(ROOT/'boundary_predictions.csv',index=False)
print(result.groupby(['seed','mode','layout']).pair.sum().unstack())
