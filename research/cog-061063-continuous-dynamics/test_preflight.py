"""Only simulator/schema/API shapes. No fitted learner or evaluated model scores."""
import json
from pathlib import Path
import numpy as np
from sim import TRAIN,DEV_SHIFT,DEV_STABLE,episode,mix,digest
from model import windows, feats
p=Path(__file__).parent/'frozen';tr=json.loads((p/'train.json').read_text());de=json.loads((p/'challenge.json').read_text())
assert set(TRAIN).isdisjoint(set(DEV_SHIFT)|set(DEV_STABLE))
assert len(tr['rows'])==len(de['rows'])==64
for split,obj in [('train',tr),('validation',de)]:
 for r in obj['rows']:
  assert not set(r['features'])-{'o','u'} and r['split']==split
 w,n,b=mix();assert np.linalg.matrix_rank(w)==4 and (np.abs(b)>.1).all() and np.linalg.matrix_rank(b)==2
p1,u,x,br=episode(2000);p2,*_=episode(2000);np.testing.assert_array_equal(p1,p2);assert np.max(np.abs(x))<1
assert windows(p1).shape==(127,48) and len(u[2:])==126
assert feats(np.zeros((10,4)),np.zeros((10,2))).shape==(10,28)
t=56;zero=np.zeros((16,2));high=zero.copy();high[:4]=.65
assert (br(t,zero)!=br(t,high)).any()
np.testing.assert_array_equal(br(t,zero),br(t,zero))
print('generator, schema, dense controls, causal windows and paired-response smoke passed; no fit')
