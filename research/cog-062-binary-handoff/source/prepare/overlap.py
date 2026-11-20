"""Freeze overlap strata using input equality only, before fitting."""
from collections import Counter,defaultdict
from pathlib import Path
import json
from prepare import feature_key,write
p=Path(__file__).parent/'data'
read=lambda n:json.loads((p/n).read_bytes())
train={feature_key(r['text']) for n in ('A.json','B.json') for r in read(n)}
e=read('E.json')
bad_groups={r['group'] for r in e if feature_key(r['text']) in train}
rows=[];c=Counter();gr=defaultdict(set)
for r in e:
    f=feature_key(r['text'])
    flags=dict(exact_tuple_overlap=f in train,group_contains_tuple_overlap=r['group'] in bad_groups)
    rows.append({k:r[k] for k in ('dialogue','group','turn')}|flags)
    for i,v in enumerate(r['delta']):
        if v not in (None,'<hold>'):
            for k,flag in (('clean_tuple',not flags['exact_tuple_overlap']),('clean_group',not flags['group_contains_tuple_overlap']),('overlap_tuple',flags['exact_tuple_overlap']),('overlap_group',flags['group_contains_tuple_overlap'])):
                if flag: c[k]+=1;c[k+('area','price')[i]]+=1;gr[k].add(r['group'])
write(p/'E-overlap.json',rows)
write(p/'overlap-preflight.json',dict(counts=dict(c),groups={k:len(v) for k,v in gr.items()},flagged_turns=sum(r['exact_tuple_overlap'] for r in rows),flagged_groups=len(bad_groups)))
print((p/'overlap-preflight.json').read_text())
