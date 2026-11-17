import unittest,torch
from data018 import build
from model import Model,tensor,ARMS
class Tests(unittest.TestCase):
 def test_pairs(self):
  data=build()
  self.assertEqual({s:sum(map(len,p.values())) for s,p in data.items()},{'train':512,'dev':576})
  for panels in data.values():
   for rows in panels.values():
    self.assertEqual(len(set(map(lambda c:len(c['text']),rows))),1)
    for a,b in zip(rows[::2],rows[1::2]):
     ea=next(e for e in a['events'] if e['op']==6);eb=next(e for e in b['events'] if e['op']==6)
     self.assertFalse(ea['gate']);self.assertTrue(eb['gate'])
     self.assertNotEqual(a['final_state'],b['final_state'])
     ca=a['text'][:ea['pos']+1].split('\n')[-1];cb=b['text'][:eb['pos']+1].split('\n')[-1]
     self.assertEqual(ca,cb);self.assertEqual(a['text'][ea['pos']+1:],b['text'][eb['pos']+1:])
 def test_reference_conditional(self):
  import random,conditional,executor
  from reference_executor import Memory
  rng=random.Random(18);m=Memory(4);st=executor.init(1)
  for _ in range(2000):
   k=rng.randrange(1,7);d=rng.randrange(4);src=rng.randrange(4);v=rng.randrange(1,5);e=rng.randrange(1,5)
   before=tuple(q.clone() for q in st)
   args=[torch.tensor([q]) for q in (k,d,src,v,e)]
   st,ans=conditional.apply(st,*args)
   if k==6:
    truth=m.read(src)==e
    if truth:m.apply('SET',d,v=v)
    else:self.assertTrue(all(torch.equal(a,b) for a,b in zip(before,st)))
   else:
    result=m.apply(['','SET','COPY','LINK','DELETE','QUERY'][k],d,j=src if k in (2,3) else None,v=v if k==1 else None)
    if k==5:self.assertEqual(int(ans[0]),0 if result is None else result)
   self.assertEqual(st[0][0].tolist(),[{'absent':0,'value':1,'ref':2}[c.kind] for c in m.cells])
   self.assertEqual(st[1][0].tolist(),[0 if c.payload is None else c.payload for c in m.cells])
 def test_split(self):
  import json
  d=build()
  def programs(panels):return {json.dumps([{k:e[k] for k in ('op','dst','src','value','expected')} for e in c['events']],sort_keys=True) for rows in panels.values() for c in rows}
  self.assertFalse(programs(d['train']) & programs(d['dev']))
 def test_labels_not_execution(self):
  rows=next(iter(build()['train'].values()))[:2];x=tensor(rows)
  for arm in ARMS:
   torch.manual_seed(18);m=Model(arm);m.eval()
   a=m.segment(x,m.init(2),return_details=True)
   fake=[{e['pos']:dict(e,src=(e['src']+1)%4,expected=4) for e in c['events']} for c in rows]
   b=m.segment(x,m.init(2),return_details=True,gold_events=fake)
   self.assertTrue(torch.equal(a[0],b[0]));self.assertTrue(torch.equal(a[1],b[1]))
   self.assertTrue(all(torch.equal(p,q) for p,q in zip(a[2][0],b[2][0])))
 def test_gradient_finite(self):
  from training import targets,loss
  rows=next(iter(build()['train'].values()))[:2]
  for arm in ARMS:
   m=Model(arm);r=m.segment(tensor(rows),m.init(2),return_details=True,gold_events=[{e['pos']:e for e in c['events']} for c in rows])
   v=loss(r[4],targets(rows),torch.ones(7),r[5]);v.backward()
   self.assertTrue(torch.isfinite(v));self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))
 def test_oracle_segment_reference(self):
  # Engineered oracle-head diagnostic, never a learned result.
  from model import Model,tensor
  for family,rows in build()['dev'].items():
   rows=rows[:2]
   for arm in ARMS:
    m=Model(arm);counter=[0];events=[{e['pos']:e for e in c['events']} for c in rows]
    def decode(h,candidates):
     t=counter[0];counter[0]+=1;out=torch.full((2,25),-100.)
     for i in range(2):
      e=events[i].get(t,dict(op=0,dst=0,src=0,value=1,expected=1,gate=False))
      for start,value in zip([0,7,11,15,19,23],[e['op'],e['dst'],e['src'],e['value']-1,e['expected']-1,int(e['gate'] or False)]):out[i,start+value]=100.
     return out
    m.decode=decode
    with torch.inference_mode():r=m.segment(tensor(rows),m.init(2),trace=True)
    for i,c in enumerate(rows):
     self.assertEqual(r[2][0][0][i].tolist(),c['final_state']['kind']);self.assertEqual(r[2][0][1][i].tolist(),c['final_state']['data'])
     for e in c['events']:
      if e['op']==5:self.assertEqual(int(r[1][i,e['pos']]),e['answer'])
 def test_invalid_conditional(self):
  import conditional,executor
  z=torch.tensor([0]);six=torch.tensor([6]);one=torch.tensor([1]);false=torch.tensor([False])
  with self.assertRaises(ValueError):conditional.apply(executor.init(1),six,z,z,z,one,false)
  with self.assertRaises(ValueError):conditional.apply(executor.init(1),six,z,torch.tensor([4]),one,one,false)
 def test_prefix_causality(self):
  rows=next(iter(build()['train'].values()))[:2];x=tensor(rows);changed=x.clone();cut=x.shape[1]//2;changed[:,cut:]=32
  for arm in ARMS:
   m=Model(arm);m.eval()
   with torch.inference_mode():a=m.segment(x,m.init(2),trace=True,return_details=True);b=m.segment(changed,m.init(2),trace=True,return_details=True)
   self.assertTrue(torch.equal(a[0][:,:cut],b[0][:,:cut]));self.assertTrue(torch.equal(a[4][:,:cut],b[4][:,:cut]))
   for p,q in zip(a[3],b[3]):self.assertTrue(torch.equal(p[:,:cut],q[:,:cut]))
 def test_models(self):
  torch.set_num_threads(1);rows=next(iter(build()['train'].values()))[:2]
  for arm in ARMS:
   m=Model(arm);r=m.segment(tensor(rows),m.init(2),gold_events=[{e['pos']:e for e in c['events']} for c in rows],return_details=True)
   self.assertEqual(r[0].shape[-1],25)
   self.assertEqual((r[-1]!=-100).sum(),2)
if __name__=='__main__':unittest.main()
