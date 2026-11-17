import torch,json,time,random,os,hashlib,zipfile
from pathlib import Path
from torch import nn
from torch.nn import functional as F
CODES=torch.tensor([0,97,98,99,100])
def carry(initial,actions):
 ids=torch.where(actions>0,torch.arange(1,actions.shape[1]+1),0).max(1).values
 return torch.cat((initial[:,None],actions),1).gather(1,ids[:,None]).squeeze(1)
class Carrier(nn.Module):
 def __init__(self,conditioned,window=3):
  super().__init__();self.conditioned=conditioned;self.window=window;self.embedding=nn.Embedding(256,32);self.frontend=nn.Linear(32*window,32);self.head=nn.Sequential(nn.Linear(64 if conditioned else 32,32 if conditioned else 64),nn.Tanh(),nn.Linear(32 if conditioned else 64,5))
 def init(self,n):return torch.zeros(n,dtype=torch.long),torch.zeros(n,self.window-1,32)
 def segment(self,x,st):
  code,hist=st;raw=self.embedding(x);context=torch.cat((hist,raw),1);windows=context.unfold(1,self.window,1).permute(0,1,3,2).flatten(2);front=self.frontend(windows).tanh()
  if self.conditioned:
   es=[]
   for t in range(x.shape[1]):
    active=self.embedding(CODES[code])*(code>0).unsqueeze(-1);e=self.head(torch.cat((front[:,t],active),-1));action=e.argmax(-1);code=torch.where(action>0,action,code);es.append(e)
   events=torch.stack(es,1)
  else:events=self.head(front);code=carry(code,events.argmax(-1))
  return events,(code,context[:,-(self.window-1):])
class GRUCarrier(nn.Module):
 def __init__(self):
  super().__init__();self.embedding=nn.Embedding(256,32);self.gru=nn.GRU(32,50,batch_first=True);self.head=nn.Linear(50,5)
 def init(self,n):return torch.zeros(n,dtype=torch.long),torch.zeros(1,n,50)
 def segment(self,x,st):
  code,h=st;y,h=self.gru(self.embedding(x),h);events=self.head(y);return events,(carry(code,events.argmax(-1)),h)
def make(arm):return GRUCarrier() if arm=='gru_parser' else Carrier(arm!='wide_mlp',3 if arm=='short_state' else 24)
def tensor(strings):return torch.tensor([list(s.encode()) for s in strings])
def records(qs):return [p+q['suffix'] for q in qs for p in q['prefix']],[v for q in qs for v in q['targets']]
def labels(qs):return torch.tensor([e+[0]*len(q['suffix']) for q in qs for e in q['events']])
def forward(m,x):
 ev,st=m.segment(x,m.init(len(x)));return ev,CODES[st[0]]
def intervention(m,arm,qs,mode):
 pre=tensor([p for q in qs for p in q['prefix']]);suf=tensor([q['suffix'] for q in qs for _ in range(2)]);a,st=m.segment(pre,m.init(len(pre)));perm=torch.arange(len(pre)).reshape(-1,2).flip(1).flatten()
 if mode=='reset':st=m.init(len(pre))
 if mode=='swap':st=(st[0][perm],st[1][:,perm] if arm=='gru_parser' else st[1][perm])
 b,st=m.segment(suf,st);return CODES[st[0]],torch.cat((a,b),1).argmax(-1)
def score(m,arm,panels,modes=('full','reset','swap')):
 rows=[]
 with torch.no_grad():
  for panel,ps in panels.items():
   for i in range(0,len(ps),32):
    qs=ps[i:i+32];strings,targets=records(qs);gold=labels(qs);allmode={mde:intervention(m,arm,qs,mde) for mde in modes};perm=torch.arange(len(strings)).reshape(-1,2).flip(1).flatten();ev,pred=forward(m,tensor(strings));assert torch.equal(pred,allmode['full'][0]);assert torch.equal(ev.argmax(-1),allmode['full'][1]);assert 'swap' not in allmode or torch.equal(allmode['swap'][0],allmode['full'][0][perm])
    for mode,(pr,ep) in allmode.items():
     for j,q in enumerate(qs):rows.append({'panel':panel,'group':q['group'],'mode':mode,'target':targets[j*2:j*2+2],'pred':pr[j*2:j*2+2].tolist(),'event_target':gold[j*2:j*2+2].tolist() if mode=='full' else [],'event_pred':ep[j*2:j*2+2].tolist() if mode=='full' else []})
 return rows
def fit(arm,seed,data,weights):
 torch.manual_seed(seed);m=make(arm);opt=torch.optim.AdamW(m.parameters(),lr=.003);rng=random.Random(seed);keys=sorted(data['train']);curve=[];token_positions=0;positive_event_labels=0;max_training_length=0;start=time.perf_counter()
 for u in range(512):
  key=rng.choice(keys);qs=[data['train'][key][rng.randrange(len(data['train'][key]))] for _ in range(8)];strings,_=records(qs);x=tensor(strings);ey=labels(qs);token_positions+=x.numel();positive_event_labels+=int((ey>0).sum());max_training_length=max(max_training_length,x.shape[1]);opt.zero_grad();ev,_=forward(m,x);loss=F.cross_entropy(ev.flatten(0,1),ey.flatten(),weight=weights);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
  if u%64==0 or u==511:curve.append([u,float(loss.detach())])
 sec=time.perf_counter()-start;m.eval();pred=score(m,arm,data['dev']);probe=score(m,arm,{k:[v[i] for i in [0,1,2,3,8,9,10,11,20,21,22,23,28,29,30,31]] for k,v in data['train'].items()},modes=('full',));times=[]
 with torch.no_grad():
  for _ in range(3):
   start=time.perf_counter();strings,_=records(data['dev']['familiar'][:4]);forward(m,tensor(strings));times.append(time.perf_counter()-start)
 return m,{'arm':arm,'seed':seed,'params':sum(p.numel() for p in m.parameters()),'updates':512,'examples':8192,'token_positions':token_positions,'positive_event_labels':positive_event_labels,'max_training_length':max_training_length,'persistent_state_bytes':sum(t.numel()*t.element_size() for t in m.init(1)),'train_seconds':sec,'curve':curve,'timings':times,'predictions':pred,'train_probe':probe}
if __name__=='__main__':
 from runtime_check import runtime_check
 runtime_check();torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);R=Path(__file__).parent;O=Path(os.environ['TRAINLAB_OUTPUT_DIR']);W=O/'retained';W.mkdir();data=json.loads((R/'data.json').read_text());weights=torch.tensor(json.loads((R/'EVENT_COUNTS.json').read_text())['weights']);results=[]
 try:
  for arm in ['short_state','wide_state','wide_mlp','gru_parser']:
   for seed in [101,202,303]:
    m,r=fit(arm,seed,data,weights);name=f'{arm}-{seed}';torch.save(m.state_dict(),W/(name+'.pt'));fresh=make(arm);fresh.load_state_dict(torch.load(W/(name+'.pt'),weights_only=True));fresh.eval();assert score(fresh,arm,data['dev'])==r['predictions'];r['checkpoint_replay_exact']=True;(W/(name+'.json')).write_text(json.dumps(r));results.append({k:v for k,v in r.items() if k not in ['predictions','train_probe']});(O/'progress.json').write_text(json.dumps(results))
  (O/'metrics.json').write_text(json.dumps({'completed_fits':12}));(O/'results.json').write_text(json.dumps(results))
  from trainlab.research_adequacy import recipe_identity
  config=json.loads((R/'recipe.json').read_text());hashes={n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in config['inputs']};obs={'fits':12,'updates':6144,'examples':98304,'minimum_params':min(x['params'] for x in results),'exact_replays':12,'paired_schedules':3}
  (O/'observations.json').write_text(json.dumps({'schema':'trainlab.execution-adequacy-observations.v1','recipe_sha256':recipe_identity(config),'inputs':hashes,'producer':{'file':'study.py','sha256':hashes['study.py']},'observations':obs}))
 finally:
  with zipfile.ZipFile(O/'evidence.zip','w',zipfile.ZIP_DEFLATED) as z:
   for p in W.glob('*'):z.write(p,p.name)
