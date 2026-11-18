from pathlib import Path
import pandas as pd
root=Path(__file__).resolve().parent
p=pd.read_csv(root/'raw.csv.gz')
idx=['seed','frame','arm','episode']
assert len(p)==3*6*4*96*2 and not p.duplicated(idx+['gold']).any()
g=p.groupby(idx)
assert g.goldslot.nunique().eq(2).all()
assert g.text.apply(lambda s:len({tuple(sorted(t.split())) for t in s})==1).all()
p['exact']=(p.pick==p.gold)&(p.slot==p.goldslot)&p.effect.eq(1)
p['abstain']=p.slot.lt(0)
assert (p.wrong==2*(p.slot.ge(0)&p.slot.ne(p.goldslot))).all()
s=p.groupby(idx).exact.all().groupby(idx[:-1]).sum().to_frame()
s=s.join(p.groupby(idx[:-1])[['wrong','abstain']].sum())
assert s.xs('oracle',level='arm').exact.eq(96).all()
s.to_csv(root/'scores.csv')
print(s.reset_index().groupby(['frame','arm']).agg(low=('exact','min'),high=('exact','max'),wrong=('wrong','sum'),abstain=('abstain','sum')))
