import torch,random,json,time,os,zipfile,hashlib
from pathlib import Path
from torch import nn
from torch.nn import functional as F
from native_csm import RecordRewrite,Config,RecordState
LETTERS='abcd'
def build_data():
 def make(seed,count,kind,gap):
  rng=random.Random(seed);pairs=[]
  for i in range(count):
   a=i%4;b=(a+1+(i//4)%3)%4
   old=rng.choice([v for v in range(4) if v not in [a,b]])
   nonce=''.join(rng.choice('jklmnopq') for _ in range(10))
   delay=(' rain sun wind. '*8)[:gap]
   suffix=delay+' The current code is '
   prefix=[]
   for v in [a,b]:
    if kind=='facts':p='Note '+nonce+'. The code is '+LETTERS[v]+'.'
    elif kind=='paraphrase':p='Note '+nonce+'. The code was '+LETTERS[old]+'. Update code to '+LETTERS[v]+'.'
    else:p='Note '+nonce+'. The code is '+LETTERS[old]+'. Now code is '+LETTERS[v]+'.'
    prefix.append(p)
   pairs.append({'group':i,'prefix':prefix,'suffix':suffix,'targets':[ord(LETTERS[v]) for v in [a,b]],'old':ord(LETTERS[old])})
  return pairs
 train={f'{k}-{gap}':make(7000+j*31+gap,256,k,gap) for j,k in enumerate(['facts','corrections']) for gap in [0,16,32]}
 dev={k:make(19000+j*101,64,k,64 if k=='long' else 32) for j,k in enumerate(['facts','corrections','long','paraphrase'])}
 # Ensure exact histories disjoint; nonce group overlap separately reported.
 tr={p+s['suffix'] for ss in train.values() for s in ss for p in s['prefix']}
 assert not any(p+s['suffix'] in tr for ss in dev.values() for s in ss for p in s['prefix'])
 return {'train':train,'dev':dev}
class GRU(nn.Module):
 def __init__(self):
  super().__init__();self.emb=nn.Embedding(256,32);self.gru=nn.GRU(32,50,batch_first=True);self.out=nn.Linear(50,256)
 def forward(self,x):h,_=self.gru(self.emb(x));return self.out(h[:,-1])
def make(arm):
 if arm in ['csm','soft']:
  m=RecordRewrite(Config(256));m.routing='soft' if arm=='soft' else 'hard';return m
 return GRU()
def tensor(strings):return torch.tensor([list(s.encode()) for s in strings])
def records(pairs):return [p+s['suffix'] for s in pairs for p in s['prefix']],[v for s in pairs for v in s['targets']]
def final(model,arm,x):
 if arm not in ['csm','soft']:return model(x)
 state=model.init_state(len(x),'cpu')
 for t in range(x.shape[1]):logits,state=model.step(x[:,t],state)
 return logits
def intervene(model,arm,pairs,mode):
 pre=tensor([p for s in pairs for p in s['prefix']]);suf=tensor([s['suffix'] for s in pairs for _ in range(2)])
 perm=torch.arange(len(pre)).reshape(-1,2).flip(1).flatten()
 if arm=='local':return model(suf)
 if arm in ['csm','soft']:
  state=model.init_state(len(pre),'cpu')
  for t in range(pre.shape[1]):_,state=model.step(pre[:,t],state)
  if mode=='reset':state=model.init_state(len(pre),'cpu');state.position=pre.shape[1]
  if mode=='swap':state=RecordState(*(getattr(state,k)[perm] for k in ['values','links','cursor','history']),state.position)
  for t in range(suf.shape[1]):logits,state=model.step(suf[:,t],state)
  return logits
 _,h=model.gru(model.emb(pre))
 if mode=='reset':h=torch.zeros_like(h)
 if mode=='swap':h=h[:,perm]
 y,_=model.gru(model.emb(suf),h);return model.out(y[:,-1])
def score(model,arm,panels):
 out=[]
 with torch.no_grad():
  for panel,pairs in panels.items():
   for k in range(0,len(pairs),8):
    ps=pairs[k:k+8];ys=torch.tensor([y for p in ps for y in p['targets']]);logits={m:intervene(model,arm,ps,m) for m in ['full','reset','swap']}
    perm=torch.arange(len(ys)).reshape(-1,2).flip(1).flatten();assert torch.equal(logits['swap'],logits['full'][perm])
    for mode,l in logits.items():
     probs=l.log_softmax(-1);pred=l.argmax(-1);restricted=l[:,97:101].argmax(-1)+97
     for j,p in enumerate(ps):
      out.append({'panel':panel,'group':p['group'],'mode':mode,'target':ys[j*2:j*2+2].tolist(),'pred':pred[j*2:j*2+2].tolist(),'restricted':restricted[j*2:j*2+2].tolist(),'nll':[-float(probs[q,ys[q]]) for q in range(j*2,j*2+2)]})
 return out
def fit(arm,seed,data):
 torch.manual_seed(seed);model=make(arm);opt=torch.optim.AdamW(model.parameters(),lr=.003);rng=random.Random(seed);keys=sorted(data['train']);curve=[];start=time.perf_counter()
 for u in range(512):
  key=rng.choice(keys);ps=[data['train'][key][rng.randrange(256)] for _ in range(8)];strings,ys=records(ps)
  if arm=='local':strings=[p['suffix'] for p in ps for _ in range(2)]
  x=tensor(strings);y=torch.tensor(ys);opt.zero_grad();loss=F.cross_entropy(final(model,arm,x),y);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
  if u%64==0 or u==511:curve.append([u,float(loss)])
 sec=time.perf_counter()-start;model.eval();pred=score(model,arm,data['dev']);probe=score(model,arm,{k:v[:16] for k,v in data['train'].items()});times=[]
 hardened=[]
 if arm=='soft':
  model.routing='hard';hardened=score(model,arm,data['dev']);model.routing='soft' 
 with torch.no_grad():
  for _ in range(3):
   a=time.perf_counter();ps=data['dev']['corrections'][:4];strings,ys=records(ps)
   if arm=='local':strings=[p['suffix'] for p in ps for _ in range(2)]
   out=final(model,arm,tensor(strings)).argmax(-1).tolist();times.append(time.perf_counter()-a)
 return model,{'arm':arm,'seed':seed,'params':sum(p.numel() for p in model.parameters()),'updates':512,'examples':8192,'train_seconds':sec,'curve':curve,'timings':times,'predictions':pred,'train_probe':probe,'hardened':hardened}
if __name__=='__main__':
 from runtime_check import runtime_check
 runtime_check();torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 R=Path(__file__).parent;O=Path(os.environ['TRAINLAB_OUTPUT_DIR']);W=O/'retained';W.mkdir();data=json.loads((R/'data.json').read_text());results=[]
 try:
  for arm in ['csm','soft','gru']:
   for seed in [101,202,303]:
    m,r=fit(arm,seed,data);name=f'{arm}-{seed}';torch.save(m.state_dict(),W/(name+'.pt'));fresh=make(arm);fresh.load_state_dict(torch.load(W/(name+'.pt'),weights_only=True));fresh.eval();assert score(fresh,arm,data['dev'])==r['predictions'];r['checkpoint_replay_exact']=True;(W/(name+'.json')).write_text(json.dumps(r));results.append({k:v for k,v in r.items() if k not in ['predictions','train_probe','hardened']});(O/'progress.json').write_text(json.dumps(results))
  (O/'metrics.json').write_text(json.dumps({'completed_fits':9}));(O/'results.json').write_text(json.dumps(results))
  from trainlab.research_adequacy import recipe_identity
  config=json.loads((R/'recipe.json').read_text());hashes={n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in config['inputs']}
  obs={'fits':9,'updates':4608,'examples':73728,'minimum_params':min(x['params'] for x in results),'exact_replays':9,'paired_schedules':3}
  (O/'observations.json').write_text(json.dumps({'schema':'trainlab.execution-adequacy-observations.v1','recipe_sha256':recipe_identity(config),'inputs':hashes,'producer':{'file':'study.py','sha256':hashes['study.py']},'observations':obs}))
 finally:
  with zipfile.ZipFile(O/'evidence.zip','w',zipfile.ZIP_DEFLATED) as z:
   for p in W.glob('*'):z.write(p,p.name)
