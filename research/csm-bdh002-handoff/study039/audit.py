from pathlib import Path
import pandas as pd

root=Path(__file__).parent
p=pd.read_csv(root/'raw.csv.gz')
ids=['seed','layout','arm','episode']
assert len(p)==3*5*4*96*4 and not p.duplicated(ids+['reforder','approval']).any()
assert p.groupby(ids).size().eq(4).all()
assert p.effect.eq(p.slot.eq(p.goldslot)).all()
assert p.wrong.eq(2*(p.slot.ge(0)&p.slot.ne(p.goldslot))).all()
p['joint']=p.ref.eq(p.gold)&p.slot.eq(p.goldslot)&p.effect.eq(1)
p['abstain']=p.slot.lt(0)
s=p.groupby(ids).joint.all().groupby(ids[:-1]).sum().to_frame()
s=s.join(p.groupby(ids[:-1])[['wrong','abstain']].sum())
assert s.xs('oracle',level='arm').joint.eq(96).all()
s.to_csv(root/'scores.csv')
print(s.reset_index().groupby(['layout','arm']).agg(low=('joint','min'),high=('joint','max'),wrong=('wrong','sum'),abstain=('abstain','sum')))
