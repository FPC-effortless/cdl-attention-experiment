from followup import *
import hashlib

def stress(seed,T):
 r=np.random.default_rng(seed);events=[];index=[[] for _ in range(32)];queries=[]
 for tx in range(T):
  k=0 if r.random()<.75 else int(r.integers(1,32));vt=int(r.integers(tx+1)) if r.random()<.3 else tx;events.append((k,vt,tx,int(r.integers(1000))))
 for _ in range(64):queries.append((0 if r.random()<.75 else int(r.integers(1,32)),int(r.integers(T)),int(r.integers(T))))
 t=time.perf_counter_ns()
 for k,vt,tx,v in events:bisect.insort(index[k],(vt,tx,v))
 prep=time.perf_counter_ns()-t;t=time.perf_counter_ns();ref=[scan(events,*q) for q in queries];sc=time.perf_counter_ns()-t;t=time.perf_counter_ns();out=[lookup(index,*q) for q in queries];ix=time.perf_counter_ns()-t
 counts=[]
 for k,vt,tx in queries:
  a=index[k];i=bisect.bisect_right(a,(vt,10**9,10**9))-1;count=0
  while i>=0:
   count+=1
   if a[i][1]<=tx:break
   i-=1
  counts.append(count)
 assert ref==out
 return dict(seed=seed,T=T,n=len(ref),exact=len(ref),prep_ns=prep,scan_ns=sc,index_ns=ix,max_candidates=max(counts),mean_candidates=float(np.mean(counts)))
if __name__=='__main__':
 x=json.loads((P/'results.json').read_text());f=json.loads((P/'followup.json').read_text());n=0
 for row in x['neural']:
  seed=row['seed'];mode=row['mode'];ck=torch.load(P/f'{mode}_{seed}.pt');model=Flat() if mode=='flat_soft' else Factor();model.load_state_dict(ck['state'])
  for i,c in enumerate(['familiar','held_pair','recombine','novel']):
   assert eval_model(model,data(seed+2000+i,1024,np.array(ck['alias']),c))==row['cohorts'][c];n+=1
 for row in x['history']:
  y=history(row['seed']);assert all(y[k]==v for k,v in row.items() if not k.endswith('_ns'))
 for row in f['acquisition']:assert acquire(row['seed'],row['active'],row['noise'],row['budget'])==row
 for row in f['compiled']:
  y=compile_test(row['seed']);assert all(y[k]==v for k,v in row.items() if k not in ['prep_ns','times'])
 z=[stress(s,T) for T in [128,2048,32768] for s in range(900,908)]
 (P/'audit.json').write_text(json.dumps(dict(neural_cohorts=n,history_seeds=len(x['history']),acquisition_records=len(f['acquisition']),compiled_seeds=len(f['compiled']),stress=z),indent=2));print('audit passed',n,len(x['history']),len(f['acquisition']),len(f['compiled']))
 for T in [128,2048,32768]:
  a=[r for r in z if r['T']==T];print(T,sum(r['exact'] for r in a),max(r['max_candidates'] for r in a),sum(r['scan_ns'] for r in a)/sum(r['index_ns']+r['prep_ns'] for r in a))
