"""Hand-authored mechanical parity with invented coefficient arrays, no fit/held data."""
import numpy as np
from serve import prepare,run,run_fused_group,observe
def packet(scope,revoke=False,tie=False):
 return {'scope':scope,'tick':4,'events':[['a',scope,2,1,[1,True]],['b',scope,2,0,[1,True]],['a',scope,5,9,['bad',True]]]+([['a',scope,4,2,[1,False]]] if revoke else []),
         'histories':{k:[[t,[float(t)]*16] for t in (2,3,4)] for k in ('a','b')}}
p=[packet('role-a'),packet('role-b',True)]*8
m=prepare({'predictive':{'w':np.arange(192,dtype=float).reshape(48,4).tolist(),'beta':[1.,-.5,.1,.7],'mean':[3.]*48,'std':[1.9]*48,'intercept':.25}})
assert set(observe(p[0]))=={'a','b'} and set(observe(p[1]))=={'b'}
baseline=[run(r,m,'predictive') for r in p]
fused,fa=run_fused_group(p,m,False); packed,fb=run_fused_group(p,m,True)
assert fused==packed==[x[0] for x in baseline]
assert fa==fb==8
assert [x['lane'] for x in packed]==['a','b']*8
print('PASS packed-vs-folded-vs-reference exact advice, same-tick revoke and stable tie fallback; hand-authored only')
