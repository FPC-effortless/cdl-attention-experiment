import torch,json,time,random,os,re,hashlib,zipfile
from pathlib import Path
from torch import nn
from torch.nn import functional as F
from baseline import make as old_make,tensor,records,final as baseline_final
from native_csm import RecordState
CODES=torch.tensor([0,97,98,99,100])
class Carrier(nn.Module):
 def __init__(self,conditioned):
  super().__init__();self.conditioned=conditioned;self.embedding=nn.Embedding(256,32);self.frontend=nn.Linear(96,32)
  self.head=nn.Sequential(nn.Linear(64 if conditioned else 32,32 if conditioned else 64),nn.Tanh(),nn.Linear(32 if conditioned else 64,5))
 def init(self,n):return (torch.zeros(n,dtype=torch.long),torch.zeros(n,2,32))
 def step(self,token,state):
  code,hist=state;raw=self.embedding(token);front=self.frontend(torch.cat((hist.flatten(1),raw),-1)).tanh()
  if self.conditioned:
   active=self.embedding(CODES[code])*(code>0).unsqueeze(-1);front=torch.cat((front,active),-1)
  event=self.head(front);action=event.argmax(-1);code=torch.where(action>0,action,code)
  return event,(code,torch.stack((hist[:,1],raw),1))
def make(arm):
 if arm.startswith('typed'):return Carrier(arm=='typed_csm')
 m=old_make('csm' if arm=='native_aux' else 'gru');m.event_head=nn.Linear(32 if arm=='native_aux' else 50,5);return m
def answer_only(m,arm,x):
 return forward(m,arm,x)[2] if arm.startswith('typed') else baseline_final(m,'csm' if arm=='native_aux' else 'gru',x).argmax(-1)
def annotate(strings):
 rows=[]
 for s in strings:
  y=[0]*len(s)
  for match in re.finditer(r'(?:is|was|to) ([abcd])\.',s):y[match.start(1)]=ord(match.group(1))-96
  rows.append(y)
 return torch.tensor(rows)
def forward(m,arm,x):
 events=[]
 if arm.startswith('typed'):
  st=m.init(len(x))
  for t in range(x.shape[1]):e,st=m.step(x[:,t],st);events.append(e)
  return torch.stack(events,1),None,CODES[st[0]]
 if arm=='native_aux':
  st=m.init_state(len(x),'cpu');outs=[]
  for t in range(x.shape[1]):
   l,st,tr=m.step(x[:,t],st,return_trace=True);outs.append(l);events.append(m.event_head(tr['payload']))
  logits=torch.stack(outs,1)
 else:hs,_=m.gru(m.emb(x));logits=m.out(hs);events=list(m.event_head(hs).unbind(1))
 return torch.stack(events,1),logits[:,-1],logits[:,-1].argmax(-1)
def intervention(m,arm,ps,mode):
 pre=tensor([p for q in ps for p in q['prefix']]);suf=tensor([q['suffix'] for q in ps for _ in range(2)]);perm=torch.arange(len(pre)).reshape(-1,2).flip(1).flatten();events=[]
 if arm.startswith('typed'):
  st=m.init(len(pre))
  for t in range(pre.shape[1]):e,st=m.step(pre[:,t],st);events.append(e.argmax(-1))
  if mode=='reset':st=m.init(len(pre))
  if mode=='swap':st=tuple(v[perm] for v in st)
  for t in range(suf.shape[1]):e,st=m.step(suf[:,t],st);events.append(e.argmax(-1))
  pred=CODES[st[0]]
 elif arm=='native_aux':
  st=m.init_state(len(pre),'cpu')
  for t in range(pre.shape[1]):
   l,st,tr=m.step(pre[:,t],st,return_trace=True);events.append(m.event_head(tr['payload']).argmax(-1))
  if mode=='reset':st=m.init_state(len(pre),'cpu');st.position=pre.shape[1]
  if mode=='swap':st=RecordState(*(getattr(st,k)[perm] for k in ['values','links','cursor','history']),st.position)
  for t in range(suf.shape[1]):
   l,st,tr=m.step(suf[:,t],st,return_trace=True);events.append(m.event_head(tr['payload']).argmax(-1))
  pred=l.argmax(-1)
 else:
  hs,h=m.gru(m.emb(pre));events.extend(m.event_head(hs).argmax(-1).unbind(1))
  if mode=='reset':h=torch.zeros_like(h)
  if mode=='swap':h=h[:,perm]
  hs,h=m.gru(m.emb(suf),h);ls=m.out(hs);events.extend(m.event_head(hs).argmax(-1).unbind(1));pred=ls[:,-1].argmax(-1)
 return pred,torch.stack(events,1)
def score(m,arm,panels):
 rows=[]
 with torch.no_grad():
  for panel,ps in panels.items():
   for i in range(0,len(ps),8):
    qs=ps[i:i+8];strings,targets=records(qs);gold=annotate(strings);allmode={mode:intervention(m,arm,qs,mode) for mode in ['full','reset','swap']};perm=torch.arange(len(strings)).reshape(-1,2).flip(1).flatten();assert torch.equal(allmode['swap'][0],allmode['full'][0][perm]);assert torch.equal(forward(m,arm,tensor(strings))[2],allmode['full'][0]);assert torch.equal(answer_only(m,arm,tensor(strings)),allmode['full'][0])
    for mode,(pred,event) in allmode.items():
     for j,q in enumerate(qs):rows.append({'panel':panel,'group':q['group'],'mode':mode,'target':targets[j*2:j*2+2],'pred':pred[j*2:j*2+2].tolist(),'event_target':gold[j*2:j*2+2].tolist() if mode=='full' else [],'event_pred':event[j*2:j*2+2].tolist() if mode=='full' else []})
 return rows
def fit(arm,seed,data,weights):
 torch.manual_seed(seed);m=make(arm);opt=torch.optim.AdamW(m.parameters(),lr=.003);rng=random.Random(seed);keys=sorted(data['train']);curve=[];start=time.perf_counter()
 for u in range(512):
  key=rng.choice(keys);qs=[data['train'][key][rng.randrange(256)] for _ in range(8)];strings,targets=records(qs);x=tensor(strings);ey=annotate(strings);opt.zero_grad();ev,answer,_=forward(m,arm,x);eloss=F.cross_entropy(ev.flatten(0,1),ey.flatten(),weight=weights);loss=eloss
  if answer is not None:loss=loss+F.cross_entropy(answer,torch.tensor(targets))
  loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
  if u%64==0 or u==511:curve.append([u,float(loss.detach()),float(eloss.detach())])
 trainsec=time.perf_counter()-start;m.eval();pred=score(m,arm,data['dev']);probe=score(m,arm,{k:v[:16] for k,v in data['train'].items()});times=[]
 with torch.no_grad():
  for _ in range(3):
   start=time.perf_counter();strings,_=records(data['dev']['corrections'][:4]);answer_only(m,arm,tensor(strings));times.append(time.perf_counter()-start)
 return m,{'arm':arm,'seed':seed,'params':sum(p.numel() for p in m.parameters()),'updates':512,'examples':8192,'train_seconds':trainsec,'curve':curve,'timings':times,'predictions':pred,'train_probe':probe}
if __name__=='__main__':
 from runtime_check import runtime_check
 runtime_check();torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);R=Path(__file__).parent;O=Path(os.environ['TRAINLAB_OUTPUT_DIR']);W=O/'retained';W.mkdir();data=json.loads((R/'data.json').read_text());weights=torch.tensor(json.loads((R/'EVENT_COUNTS.json').read_text())['weights']);results=[]
 try:
  for arm in ['typed_csm','typed_mlp','native_aux','gru_aux']:
   for seed in [101,202,303]:
    m,r=fit(arm,seed,data,weights);name=f'{arm}-{seed}';torch.save(m.state_dict(),W/(name+'.pt'));fresh=make(arm);fresh.load_state_dict(torch.load(W/(name+'.pt'),weights_only=True));fresh.eval();assert score(fresh,arm,data['dev'])==r['predictions'];r['checkpoint_replay_exact']=True;(W/(name+'.json')).write_text(json.dumps(r));results.append({k:v for k,v in r.items() if k not in ['predictions','train_probe']});(O/'progress.json').write_text(json.dumps(results))
  (O/'metrics.json').write_text(json.dumps({'completed_fits':12}));(O/'results.json').write_text(json.dumps(results))
  from trainlab.research_adequacy import recipe_identity
  config=json.loads((R/'recipe.json').read_text());hashes={n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in config['inputs']};obs={'fits':12,'updates':6144,'examples':98304,'minimum_params':min(x['params'] for x in results),'exact_replays':12,'paired_schedules':3}
  (O/'observations.json').write_text(json.dumps({'schema':'trainlab.execution-adequacy-observations.v1','recipe_sha256':recipe_identity(config),'inputs':hashes,'producer':{'file':'study.py','sha256':hashes['study.py']},'observations':obs}))
 finally:
  with zipfile.ZipFile(O/'evidence.zip','w',zipfile.ZIP_DEFLATED) as z:
   for p in W.glob('*'):z.write(p,p.name)
