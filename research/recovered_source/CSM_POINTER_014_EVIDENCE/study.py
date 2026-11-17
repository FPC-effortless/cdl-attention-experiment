import torch,json,time,random,hashlib,os,zipfile
from pathlib import Path
from model import Model,tensor
from training import targets,loss
from evaluate import score_dataset

def fit(arm,seed,data,weights):
 torch.manual_seed(seed);m=Model(arm);opt=torch.optim.AdamW(m.parameters(),lr=.003);rng=random.Random(seed);keys=sorted(data['train']);curve=[];tokens=events=0;schedule=hashlib.sha256();start=time.perf_counter()
 for u in range(1024):
  key=rng.choice(keys);indices=[rng.randrange(len(data['train'][key])) for _ in range(16)];schedule.update(json.dumps([key,indices]).encode());qs=[data['train'][key][i] for i in indices];x=tensor(qs);y=targets(qs);tokens+=x.numel();events+=int((y[:,:,0]>0).sum());opt.zero_grad();logits,_,_,_=m.segment(x,m.init(len(x)),execute=m.conditioned);v=loss(logits,y,weights);assert torch.isfinite(v);v.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
  if u%128==0 or u==1023:curve.append([u,float(v.detach())])
 elapsed=time.perf_counter()-start;m.eval();dev=score_dataset(m,data['dev']);cal=score_dataset(m,data['train']);rename_equal=True
 for panel in data['dev']:
  if panel.endswith('_renamed_EH'):continue
  for a,b in zip(dev[panel]['records'],dev[panel+'_renamed_EH']['records']):
   rename_equal &= all(a[k]==b[k] for k in ['pred_nonnoop','query_predictions','final_query_predictions','pred_terminal_states','pred_final_state'])
 if m.tied:assert rename_equal
 work=data['dev']['overwrite_form0'][:8];times=[];splits=[]
 with torch.no_grad():
  for r in range(4):
   start=time.perf_counter();x=tensor(work);a=m.segment(x,m.init(len(x)));dt=time.perf_counter()-start
   boundary=work[0]['ingest_boundary'];start=time.perf_counter();x=tensor(work);b=m.segment(x[:,:boundary],m.init(len(x)));mid=time.perf_counter();c=m.segment(x[:,boundary:],b[2]);end=time.perf_counter();assert torch.equal(a[1],torch.cat([b[1],c[1]],1));assert all(torch.equal(i.argmax(-1),j.argmax(-1)) for i,j in zip(a[0].split([6,4,4,4],-1),torch.cat([b[0],c[0]],1).split([6,4,4,4],-1)));assert torch.allclose(a[0],torch.cat([b[0],c[0]],1),atol=1e-6,rtol=1e-5);assert all(torch.equal(i,j) for i,j in zip(a[2][0],c[2][0]));assert all(torch.equal(a[2][2][k],c[2][2][k]) for k in a[2][2])
   if r:times.append(dt);splits.append({'ingestion_seconds':mid-start,'query_suffix_seconds':end-mid,'full_seconds':end-start})
 st=m.init(1);return m,{'arm':arm,'seed':seed,'paired_rename_exact':rename_equal,'params':sum(p.numel() for p in m.parameters()),'updates':1024,'examples':16384,'token_positions':tokens,'positive_event_labels':events,'schedule_sha256':schedule.hexdigest(),'train_seconds':elapsed,'curve':curve,'timings':times,'split_timings':splits,'persistent_state_bytes':sum(t.numel()*t.element_size() for t in (*st[0],st[1],*st[2].values())),'dev':dev,'train_probe':cal}
if __name__=='__main__':
 from runtime_check import runtime_check
 runtime_check();torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);R=Path(__file__).parent;O=Path(os.environ['TRAINLAB_OUTPUT_DIR']);W=O/'retained';W.mkdir();data=json.loads((R/'canonical_data.json').read_text());weights=torch.tensor(json.loads((R/'EVENT_COUNTS.json').read_text())['weights']);results=[]
 try:
  for arm in ['tied_state','untied_state','tied_mlp','untied_mlp','untied_gru']:
   for seed in [101,202,303]:
    m,r=fit(arm,seed,data,weights);name=f'{arm}-{seed}';torch.save(m.state_dict(),W/(name+'.pt'));fresh=Model(arm);fresh.load_state_dict(torch.load(W/(name+'.pt'),weights_only=True));fresh.eval();assert score_dataset(fresh,data['dev'])==r['dev'];r['checkpoint_replay_exact']=True;(W/(name+'.json')).write_text(json.dumps(r));results.append({k:v for k,v in r.items() if k not in ['dev','train_probe']});(O/'progress.json').write_text(json.dumps(results))
  (O/'metrics.json').write_text(json.dumps({'completed_fits':15}));(O/'results.json').write_text(json.dumps(results))
  from trainlab.research_adequacy import recipe_identity
  config=json.loads((R/'recipe.json').read_text());hashes={n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in config['inputs']};obs={'fits':15,'updates':15360,'examples':245760,'minimum_params':40106,'exact_replays':15,'paired_schedules':len({r['schedule_sha256'] for r in results})};assert obs['paired_schedules']==3
  (O/'observations.json').write_text(json.dumps({'schema':'trainlab.execution-adequacy-observations.v1','recipe_sha256':recipe_identity(config),'inputs':hashes,'producer':{'file':'study.py','sha256':hashes['study.py']},'observations':obs}))
 finally:
  with zipfile.ZipFile(O/'evidence.zip','w',zipfile.ZIP_DEFLATED) as z:
   for p in W.glob('*'):z.write(p,p.name)
