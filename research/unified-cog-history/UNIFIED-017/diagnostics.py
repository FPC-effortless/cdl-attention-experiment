from run import *
import pickle,statistics,io,itertools

def load(path):
 c=torch.load(path,weights_only=True);m=Net(c['arm'],c['ff']);m.load_state_dict(c['weights']);m.eval();return m,c

@torch.no_grad()
def mutation(m,seed,threshold):
 out={k:{p:{} for p in ('native','store','calibrated')} for k in ('value','ambiguity','nuisance')}
 for i in range(8):
  x,a,k,v,y,target=batch(25000000+seed*100+i,8,128,'both',True);ii=torch.arange(len(y));base_z,base_h=m(x);base_p=decode(base_h);base_s,_=tensor_store(base_p);base_c,_=tensor_store(base_p,threshold);original={'native':base_z.argmax(-1),'store':base_s,'calibrated':base_c}
  for name in out:
   xx=x.clone();yy=y.clone()
   if name=='value':
    pos=(xx[:,0]>=12)&(xx[:,0]<20);xx[:,0]=torch.where(pos,((v[:,0]+1)%V+12)[:,None],xx[:,0]);yy[ii,target]=(v[:,0]+1)%V
   elif name=='ambiguity':
    xx[:,0]=torch.where(xx[:,0]==SURE,MAYBE,xx[:,0]);yy[ii,target]=8
   else:
    # same number of words; replace only distractor in first event; never use target as distractor
    pos=(xx[:,0]==NOT).long().argmax(-1);old=xx[ii,0,pos+2];new=(old+1)%N;new=torch.where(new==target,(new+1)%N,new);xx[ii,0,pos+1]=(3-new)%4+8;xx[ii,0,pos+2]=new
   z,h=m(xx);p=decode(h);s,_=tensor_store(p);cs,_=tensor_store(p,threshold);new={'native':z.argmax(-1),'store':s,'calibrated':cs}
   for mode in new:
    b=original[mode];n=new[mode];mask=torch.ones_like(b,dtype=torch.bool);mask[ii,target]=False;merge(out[name][mode],dict(n=len(y),base=int((b==y).all(1).sum()),mutated=int((n==yy).all(1).sum()),pair=int(((b==y).all(1)&(n==yy).all(1)).sum()),target_pair=int(((b[ii,target]==y[ii,target])&(n[ii,target]==yy[ii,target])).sum()),unrelated_same=int(((b==n)|~mask).all(1).sum())))
 return out

def verify(path):
 torch.set_num_threads(1);m,c=load(path);r=json.loads(path.with_suffix('.json').read_text());assert hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256'];th,dev=calibration(m,c['seed']);ev=evaluate(m,c['seed'],th);ok=ev==r['eval'] and dev==r['calibration'] and th==r['threshold'];mut=mutation(m,c['seed'],th) if c['regime']=='joint' else None;print('REPLAY',path.stem,ok,flush=True);return dict(fit=path.stem,hash=r['sha256'],replay_exact=ok,mutation=mut)

def checked_commit(s,act,key,value,conf,threshold=.7):
 if not isinstance(act,int) or act not in (0,1,2,3):return s.copy(),False
 if not isinstance(key,int) or not 0<=key<N:return s.copy(),False
 if not isinstance(value,int) or not 0<=value<V:return s.copy(),False
 if not isinstance(conf,(int,float)) or not math.isfinite(conf) or not 0<=conf<=1 or conf<threshold:return s.copy(),False
 if act>=2:return s.copy(),False
 out=s.copy();out[key]=value if act==0 else 8;return out,True

def schema():
 n=bad=changes=0;s=[i for i in range(N)];
 for a,k,v,c in itertools.product(range(4),range(-1,N+1),range(-1,V+1),[0.,.69,.7,1.,1.1,float('nan')]):
  q,ok=checked_commit(s,a,k,v,c);valid=a<2 and 0<=k<N and 0<=v<V and math.isfinite(c) and .7<=c<=1;expect=s.copy()
  if valid:expect[k]=v if a==0 else 8
  bad+=int(q!=expect or ok!=valid or checked_commit(q,a,k,v,c)[0]!=q);n+=1;changes+=int(ok)
 return dict(tests=n,violations=bad,accepted=changes)

@torch.no_grad()
def timing(path):
 m,c=load(path);out=[]
 for t in [16,128,1024]:
  for b in [1,8]:
   x,a,k,v,y,target=batch(26000000+c['seed']*100+t+b,b,t,'both',t>16);half=t//2
   # Includes interpretation of first and second chunks and serialization/restore.
   def one(mode):
    p1=decode(m.encode(x[:,:half])[1]);p2=decode(m.encode(x[:,half:])[1]);
    if mode=='tensor':
     s,_=tensor_store(p1);blob=s.numpy().tobytes();st=torch.frombuffer(bytearray(blob),dtype=torch.int64).reshape_as(s);f,_=tensor_store(p2,initial=st)
    else:
     _,ds=dict_store(p1);blob=json.dumps(ds,separators=(',',':')).encode();st=[{int(k):int(v) for k,v in d.items()} for d in json.loads(blob)];f,_=dict_store(p2,initial=st)
    return f,len(blob)
   a0,n0=one('tensor');b0,n1=one('dict');assert torch.equal(a0,b0);stats={m:[] for m in ('tensor','dict')}
   for rep in range(3):
    order=('tensor','dict') if (rep+c['seed'])%2 else ('dict','tensor')
    for mode in order:
     st=time.perf_counter();f,bl=one(mode);stats[mode].append(time.perf_counter()-st);assert torch.equal(f,a0)
   out.append(dict(fit=path.stem,horizon=t,batch=b,correct=int((a0==y).all(1).sum()),histories=b,bytes_tensor=n0,bytes_dict=n1,seconds=stats))
 return out
if __name__=='__main__':
 torch.set_num_threads(1);torch.set_num_interop_threads(1);paths=sorted((P/'fits').glob('*.pt'));assert len(paths)==18,len(paths)
 with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
  replays=list(pool.map(verify,paths))
 (P/'verification.json').write_text(json.dumps(dict(replays=replays,schema=schema()),indent=2));assert all(x['replay_exact'] for x in replays)
 timings=[]
 for p in paths:
  if '_joint_' in p.name:timings.extend(timing(p));print('TIMED',p.stem,flush=True)
 (P/'timing.json').write_text(json.dumps(timings,indent=2))
