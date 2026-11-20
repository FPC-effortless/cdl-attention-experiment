import hashlib,json,os,sqlite3
from pathlib import Path
import numpy as np
from upstream.belief import DialogueState, EMPTY,HOLD
UNKNOWN='<unknown>'
class Runtime:
 def __init__(self,labels,gates,arm,binding):
  self.arm,self.binding=arm,binding;self.age=[0]*len(labels)
  self.r=DialogueState(labels,gates,method='soft' if arm=='soft' else 'verified' if arm=='verified' else 'hard',power=2,hold_weight=.25)
 def observe(self,turn,scores):
  patch=self.r.observe(turn,scores)
  self.age=[0 if labels[int(p.argmax())]!=HOLD else a+1 for a,p,labels in zip(self.age,scores,self.r.labels.values())]
  vals=[x[0] for x in self.r.estimates().values()]
  if self.arm=='noop': vals=[EMPTY]*len(vals)
  if self.arm=='reset':vals=[ls[int(p.argmax())] if ls[int(p.argmax())]!=HOLD else UNKNOWN for p,ls in zip(scores,self.r.labels.values())]
  if self.arm.startswith('expiry'):
   ttl=int(self.arm[-1]); verified=self.r.verified()
   vals=[UNKNOWN if a>=ttl and v!=verified.get(k,EMPTY) else v for (k,_),a,v in zip(self.r.labels.items(),self.age,vals)]
  return {'patch':patch,'verified':self.r.verified(),'read':vals}
 def dump(self):
  return {'binding':self.binding,'arm':self.arm,'labels':self.r.labels,'cutoffs':self.r.cutoffs,'age':self.age,'last':self.r.last_turn,'mass':[x.tolist() for x in self.r.mass],'sql':'\n'.join(self.r._store.db.iterdump())}
 def save(self,path):
  d=self.dump();body=json.dumps(d,sort_keys=True).encode();payload=json.dumps({'body':d,'sha256':hashlib.sha256(body).hexdigest()});tmp=Path(str(path)+'.tmp')
  with tmp.open('w') as f:f.write(payload);f.flush();os.fsync(f.fileno())
  os.replace(tmp,path)
 @classmethod
 def restore(cls,path,binding,arm):
  p=json.loads(Path(path).read_text());d=p['body']
  if hashlib.sha256(json.dumps(d,sort_keys=True).encode()).hexdigest()!=p['sha256'] or d['binding']!=binding or d['arm']!=arm:raise ValueError('snapshot identity/checksum mismatch')
  r=cls(d['labels'],d['cutoffs'],arm,binding);r.r._store.db.close();r.r._store.db=sqlite3.connect(':memory:');r.r._store.db.executescript(d['sql']);r.r.mass=[np.array(x) for x in d['mass']];r.r.last_turn=d['last'];r.age=d['age'];return r
 def close(self):self.r.close()
