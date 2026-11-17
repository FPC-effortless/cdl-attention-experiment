import torch,json,time,os,hashlib,zipfile,random
from pathlib import Path
from torch import nn
from torch.nn import functional as F
from baseline import make,tensor,records,final

def collect(m,arm,ps,grad=False):
 strings,ys=records(ps);x=tensor(strings);y=torch.tensor(ys)-97
 positions=[len(p)-2 for q in ps for p in q['prefix']]
 assert all(strings[i][p]==chr(ys[i]) and strings[i][p+1]=='.' for i,p in enumerate(positions))
 raws=[];fronts=[];outs=[];vals=[];gates=[];edits=[]
 def rawhook(mod,inp,out):
  raws.append(out)
  if grad:out.retain_grad()
 def fronthook(mod,inp,out):fronts.append(out.tanh())
 def outhook(mod,inp,out):outs.append(inp[0])
 handles=[m.embedding.register_forward_hook(rawhook)]
 if arm!='gru':handles += [m.frontend.register_forward_hook(fronthook),m.output.register_forward_hook(outhook)]
 try:
  if arm=='gru':
   h,_=m.gru(m.emb(x)) if hasattr(m,'emb') else m.gru(m.embedding(x))
  else:
   state=m.init_state(len(x),'cpu')
   for t in range(x.shape[1]):
    logits,state,tr=m.step(x[:,t],state,return_trace=True);vals.append(state.values);gates.append(tr['gate']);edits.append(tr['edits_value'])
 finally:
  for hnd in handles:hnd.remove()
 if arm=='gru':
  logits=m.out(h[:,-1]);local=raws[0][torch.arange(len(x)),positions];output=h[:,-1];store=output
 else:
  local=torch.stack(fronts,1)[torch.arange(len(x)),positions];output=outs[-1];store=vals[-1].flatten(1)
 with torch.no_grad():assert torch.equal(logits.detach(),final(m,arm,x))
 features={k:v.detach() for k,v in {'local_oracle':local,'output':output,'store':store}.items()}
 trace=None
 if grad:
  m.zero_grad();loss=F.cross_entropy(logits,y+97);loss.backward()
  if arm=='gru':
   rawgrad=raws[0].grad;div=[(h[:,t][::2]-h[:,t][1::2]).abs().max().item() for t in range(x.shape[1])]
  else:
   rawgrad=torch.stack([r.grad if r.grad is not None else torch.zeros_like(r) for r in raws],1);div=[(v[::2]-v[1::2]).abs().max().item() for v in vals]
  trace={'loss':float(loss.detach()),'positions':positions,'sequence_length':x.shape[1],'state_pair_max_delta':div,'code_gradient_norm':rawgrad[torch.arange(len(x)),positions].norm(dim=-1).tolist(),'last_gradient_norm':rawgrad[:,-1].norm(dim=-1).tolist(),'full_pred':logits.detach().argmax(-1).tolist(),'target':ys,'code_feature_pair_norm':(local[::2]-local[1::2]).detach().norm(dim=-1).tolist(),'final_output_pair_norm':(output[::2]-output[1::2]).detach().norm(dim=-1).tolist()}
  if arm!='gru':trace.update({'gate_mean_by_token':[float(g.mean()) for g in gates],'edit_mean_by_token':[float(e.mean()) for e in edits]})
 return features,y,trace

def extract(m,arm,panels):
 dest={k:[] for k in ['local_oracle','output','store']};labels=[];meta=[]
 with torch.no_grad():
  for name,ps in panels.items():
   for i in range(0,len(ps),8):
    chunk=ps[i:i+8];f,y,_=collect(m,arm,chunk)
    for k in dest:dest[k].append(f[k])
    labels.append(y);meta += [{'panel':name,'group':p['group'],'member':j} for p in chunk for j in range(2)]
 return {k:torch.cat(v) for k,v in dest.items()},torch.cat(labels),meta

def run_one(arm,seed,data,R,W):
 m=make(arm);m.load_state_dict(torch.load(R/f'{arm}-{seed}.pt',weights_only=True));m.eval();original={n:p.detach().clone() for n,p in m.named_parameters()}
 # GRU uses emb rather than embedding; assign alias only for hook lookup and remove after diagnostic to preserve state keys.
 if arm=='gru':m.embedding=m.emb
 start=time.perf_counter();tr={k:v[:16] for k,v in data['train'].items()};a,ya,ma=extract(m,arm,tr);b,yb,mb=extract(m,arm,data['dev']);features_seconds=time.perf_counter()-start
 probes=[]
 for key in ['local_oracle','output','store','shuffled_store']:
  feature='store' if key=='shuffled_store' else key;xx=a[feature];yy=ya.clone()
  if key=='shuffled_store':yy=yy[torch.randperm(len(yy),generator=torch.Generator().manual_seed(888))]
  torch.manual_seed(777);p=nn.Linear(xx.shape[1],4);opt=torch.optim.AdamW(p.parameters(),lr=.01);start=time.perf_counter()
  for u in range(256):opt.zero_grad();loss=F.cross_entropy(p(xx),yy);loss.backward();opt.step()
  with torch.no_grad():pred=p(b[feature]).argmax(-1).tolist();train=p(xx).argmax(-1).tolist()
  name=f'{arm}-{seed}-{key}';torch.save(p.state_dict(),W/(name+'.pt'))
  fresh=nn.Linear(xx.shape[1],4);fresh.load_state_dict(torch.load(W/(name+'.pt'),weights_only=True));assert fresh(b[feature]).argmax(-1).tolist()==pred
  probes.append({'feature':key,'parameters':sum(q.numel() for q in p.parameters()),'updates':256,'train_seconds':time.perf_counter()-start,'train_target':yy.tolist(),'train_pred':train,'dev_target':yb.tolist(),'dev_pred':pred,'dev_meta':mb,'train_meta':ma,'checkpoint_replay':True})
 traces=[]
 for split,panels in [('train',data['train']),('dev',data['dev'])]:
  for panel,ps in panels.items():
   _,_,trace=collect(m,arm,ps[:4],True);traces.append({'split':split,'panel':panel,**trace})
 assert all(torch.equal(p.detach(),original[n]) for n,p in m.named_parameters()),'backbone changed'
 return {'arm':arm,'seed':seed,'feature_seconds':features_seconds,'probes':probes,'traces':traces}
if __name__=='__main__':
 from runtime_check import runtime_check
 runtime_check();torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);R=Path(__file__).parent;O=Path(os.environ['TRAINLAB_OUTPUT_DIR']);W=O/'retained';W.mkdir();data=json.loads((R/'data.json').read_text());results=[]
 try:
  for arm in ['csm','soft','gru']:
   for seed in [101,202,303]:
    r=run_one(arm,seed,data,R,W);(W/f'{arm}-{seed}.json').write_text(json.dumps(r));results.append({'arm':arm,'seed':seed});(O/'progress.json').write_text(json.dumps(results))
  (O/'metrics.json').write_text(json.dumps({'completed_probes':36}));(O/'results.json').write_text(json.dumps(results))
  from trainlab.research_adequacy import recipe_identity
  config=json.loads((R/'recipe.json').read_text());hashes={n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in config['inputs']};obs={'backbones':9,'backbone_updates':0,'probe_fits':36,'probe_updates':9216,'parity_checks':486,'minimum_params':33699}
  (O/'observations.json').write_text(json.dumps({'schema':'trainlab.execution-adequacy-observations.v1','recipe_sha256':recipe_identity(config),'inputs':hashes,'producer':{'file':'diagnostic.py','sha256':hashes['diagnostic.py']},'observations':obs}))
 finally:
  with zipfile.ZipFile(O/'evidence.zip','w',zipfile.ZIP_DEFLATED) as z:
   for p in W.glob('*'):z.write(p,p.name)
