import json, pathlib, numpy as np
p=pathlib.Path(__file__).resolve().parent/'frozen'
tr=json.loads((p/'train.json').read_text());de=json.loads((p/'challenge.json').read_text());truth=json.loads((p/'truth.json').read_text())
assert len(tr['rows'])==len(de['rows'])==64
assert len({r['id'] for r in tr['rows']})==64 and len({r['id'] for r in de['rows']})==64
assert {r['id'] for r in tr['rows']}.isdisjoint(r['id'] for r in de['rows'])
for rows,train in [(tr['rows'],True),(de['rows'],False)]:
 for row in rows:
  seed=int(row['id']); n=128 if train else 80
  mixrng=np.random.default_rng(61063); M=np.linalg.qr(mixrng.normal(size=(16,7)))[0];B=mixrng.uniform(.17,.32,(4,2))*np.array([[1,-1],[1,1],[-1,1],[1,-1]])
  r=np.random.default_rng(seed);A=np.diag([.55,.48,.61,.47])+r.normal(0,.032,(4,4));np.fill_diagonal(A,[.55,.48,.61,.47]);A[1,0]+=.17;A[2,1]-=.19;A[3,2]+=.13
  drift=np.zeros((4,4));drift[1,0]=-.39;drift[2,1]=.36;drift[3,0]=.25
  u=r.uniform(-.8,.8,(n,2));e=r.normal(0,.014,(n,4));q=r.normal(0,.13,(n,3));noise=r.normal(0,.014,(n+1,16))
  x=np.zeros((n+1,4));v=np.zeros((n+1,3));x[0]=r.uniform(-.45,.45,4);v[0]=r.normal(0,.35,3)
  def evolve(y,t,act):
   mat=A+(drift if t>=40 and (train or seed<3000) else 0)
   return np.tanh(mat@y+B@act+np.array([.11*y[1]*y[2],.18*y[0]*act[0],.16*act[0]*act[1],-.12*y[0]*y[3]])+e[t])
  for t in range(n):x[t+1]=evolve(x[t],t,u[t]);v[t+1]=.92*v[t]+q[t]
  obs=x@M[:,:4].T+1.65*v@M[:,4:].T+noise
  k=129 if train else 57
  np.testing.assert_array_equal(obs[:k],row['features']['o']);np.testing.assert_array_equal(u[:k-1],row['features']['u'])
  if train:np.testing.assert_array_equal(x[::4],truth['calibration'][row['id']])
  else:
   g=truth['evaluation'][row['id']];assert g['shifted']==(seed<3000)
   for support in [0,4,16]:
    t=40+support;check=g['checks'][str(support)];np.testing.assert_array_equal(x[t],check['z'])
    for b,levels in zip(check['branches'],[(-1,-1),(1,-1),(-1,1),(1,1)]):
     plan=np.zeros((16,2));plan[:4]=np.array(levels)*.65;np.testing.assert_array_equal(plan,b['u']);y=x[t].copy();out=[]
     for j,act in enumerate(plan):y=evolve(y,t+j,act);out.append(y@M[:,:4].T+1.65*v[t+j+1]@M[:,4:].T+noise[t+j+1])
     np.testing.assert_array_equal(out,b['y'])
print('PASS: 128 complete source episodes, split IDs, all 768 four-step branch plans/truth, latent calibration/state matched independently; no model fit or scoring')
