from pathlib import Path
import pandas as pd

root=Path(__file__).parent
p=pd.read_csv(root/'raw.csv.gz',keep_default_na=False,dtype={'perm':str})
ids=['seed','mode','layout','perm','episode']
assert len(p)==3*3*4*6*96*2 and not p.duplicated(ids+['query']).any()
pairs=p.groupby(ids)
assert pairs.size().eq(2).all() and pairs.gold_slot.nunique().eq(2).all()
assert pairs.text.apply(lambda ts:len({tuple(sorted(t.split())) for t in ts})==1).all()
p['joint']=p.gold_key.eq(p.key)&p.gold_slot.eq(p.slot)&p.effects.eq(1)
p['abstain']=p.slot.lt(0)
assert p.effects.eq(p.gold_slot.eq(p.slot)).all()
assert p.wrong_commits.eq(2*(p.slot.ge(0)&p.slot.ne(p.gold_slot))).all()
s=p.groupby(ids).joint.all().groupby(ids[:-1]).sum().to_frame('pairs')
s=s.join(p.groupby(ids[:-1])[['wrong_commits','abstain']].sum())
assert s.xs('oracle',level='mode').pairs.eq(96).all()
s.to_csv(root/'scores.csv')
print(s.reset_index().groupby(['mode','layout']).agg(minimum=('pairs','min'),maximum=('pairs','max'),wrong=('wrong_commits','sum'),abstain=('abstain','sum')))
