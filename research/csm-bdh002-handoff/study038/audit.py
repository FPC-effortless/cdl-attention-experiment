from pathlib import Path
import pandas as pd
root = Path(__file__).parent
p = pd.read_csv(root/'raw.csv.gz')
ids = ['seed','frame','arm','episode']
assert len(p) == 3*6*5*96*4
assert not p.duplicated(ids+['reforder','approval']).any()
assert p.groupby(ids).size().eq(4).all()
assert p.gold.eq(p.reforder ^ p.approval).all()
assert p.effect.eq(p.slot.eq(p.goldslot)).all()
assert p.wrong.eq(2*(p.slot.ge(0)&p.slot.ne(p.goldslot))).all()
p['joint'] = p.gold.eq(p.pick)&p.slot.eq(p.goldslot)&p.effect.eq(1)
p['abstained'] = p.slot.lt(0)
s = p.groupby(ids).joint.all().groupby(ids[:-1]).sum().to_frame()
s = s.join(p.groupby(ids[:-1])[['wrong','abstained']].sum())
assert s.xs('oracle',level='arm').joint.eq(96).all()
s.to_csv(root/'scores.csv')
print(s.xs(11).to_string())
