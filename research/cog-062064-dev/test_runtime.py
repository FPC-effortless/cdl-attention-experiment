import tempfile,json
from pathlib import Path
import numpy as np
from runtime import Runtime,UNKNOWN
labels={'hotel-area':['<empty>','<hold>','north'],'hotel-pricerange':['<empty>','<hold>','cheap']};gates={k:.99 for k in labels}
with tempfile.TemporaryDirectory() as t:
 r=Runtime(labels,gates,'expiry2','x');p=[np.array([.05,.05,.9])]*2;h=[np.array([.01,.98,.01])]*2
 assert r.observe(0,p)['read']==['north','cheap'];assert r.observe(2,h)['read']==['north','cheap'];assert r.observe(4,h)['read']==[UNKNOWN]*2
 path=Path(t)/'x';r.save(path);b=Runtime.restore(path,'x','expiry2');assert b.dump()==r.dump()
 assert b.observe(6,p)==r.observe(6,p);assert b.dump()==r.dump()
 for identity,arm in [('other','expiry2'),('x','expiry4')]:
  try:Runtime.restore(path,identity,arm);assert False
  except ValueError:pass
 corrupt=json.loads(path.read_text());corrupt['body']['last']=100;path.write_text(json.dumps(corrupt))
 try:Runtime.restore(path,'x','expiry2');assert False
 except ValueError:pass
 r.close();b.close()
print('TTL boundaries, refresh, snapshot full parity, policy/source identity, corrupt rejection passed')
