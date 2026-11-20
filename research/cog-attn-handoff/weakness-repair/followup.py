from run import *
from collections import defaultdict
# Fresh observations: a meaning is either overwrite or add, seen entity aliases are routed by learned modules.
# If destination is nonzero, the two possible results differ; intervene on destination to disambiguate.
def acquire(seed,active,noise,budget):
 r=np.random.default_rng(seed);ok=admit=correct=0;observations=0
 for truth in r.integers(0,2,128):
  logodds=0.;n=0
  for i in range(budget):
   s=int(r.integers(4));d=int(r.integers(1,4) if active else r.integers(4));y=[s,(s+d)%4]
   n+=1
   if y[0]!=y[1]:
    reported=int(truth)^(int(r.random()<noise));logodds+=(1 if reported else -1)*(100 if noise==0 else np.log((1-noise)/noise))
   if abs(logodds)>=np.log(19):break
  pred=int(logodds>0);ok+=pred==truth;admit+=abs(logodds)>=np.log(19);correct+=(pred==truth and abs(logodds)>=np.log(19));observations+=n
 return dict(seed=seed,active=active,noise=noise,budget=budget,n=128,correct=int(ok),admit=int(admit),admit_correct=int(correct),observations=observations)
# Learned factor checkpoint compiled for exact deterministic decisions on seen cue combinations.
def predict_compiled(a,m,v,aliases,roles,verbs):
 x=aliases[a[:,0]];y=aliases[a[:,1]];rev=roles[m[:,0]]<roles[m[:,1]]
 return verbs[v]*N*N+np.where(rev,y,x)*N+np.where(rev,x,y)
def compile_test(seed):
 ck=torch.load(P/f'factor_soft_{seed}.pt');model=Factor();model.load_state_dict(ck['state']);d=data(seed+6000,4096,np.array(ck['alias']),'recombine');a,m,v,st,y,g,mask=d
 t=time.perf_counter_ns()
 with torch.no_grad(): al=model.alias.weight.argmax(-1).numpy().copy();roles=model.role.weight[:,0].numpy().copy();verbs=model.verb.weight.argmax(-1).numpy().copy()
 prep=time.perf_counter_ns()-t
 with torch.no_grad(): ref=model(a,m,v).argmax(1).numpy()
 out=predict_compiled(a.numpy(),m.numpy(),v.numpy(),al,roles,verbs);assert np.array_equal(ref,out)
 # Clarification on missing both markers and operation: enumerate four grounded actions, ask iff divergent.
 x=al[a[:,0].numpy()];z=al[a[:,1].numpy()];stn=st.numpy();truthsrc=src[g.numpy()];truthdst=dst[g.numpy()]
 def ambiguous(x,z):
  alternatives=[]
  for o in range(2):
   for i,j in [(x,z),(z,x)]:
    u=stn.copy();u[np.arange(len(g)),j]=stn[np.arange(len(g)),i] if o==0 else (stn[np.arange(len(g)),i]+stn[np.arange(len(g)),j])%4;alternatives.append(u)
  return np.any(np.stack(alternatives)!=alternatives[0],axis=(0,2))
 amb=ambiguous(truthsrc,truthdst);ask=ambiguous(x,z)
 # Simulated answer to clarification supplies intended direction and operation, learned entity binding stays in use.
 # Not learned language generation or spontaneous ambiguity detection.
 # Choose an arbitrary legal interpretation when all predicted consequences coincide.
 default=x*N+z
 resolved=np.where(ask,out,default);exact=(mask.numpy()[np.arange(len(g)),resolved]).sum()
 default_wrong=int((~mask.numpy()[np.arange(len(g)),default]).sum())
 times={}
 for b in [1,256]:
  aa,mm,vv=a[:b],m[:b],v[:b];an,mn,vn=aa.numpy(),mm.numpy(),vv.numpy();ts=[];cs=[]
  with torch.no_grad():
   for _ in range(300):
    t=time.perf_counter_ns();model(aa,mm,vv).argmax(1);ts.append(time.perf_counter_ns()-t);t=time.perf_counter_ns();predict_compiled(an,mn,vn,al,roles,verbs);cs.append(time.perf_counter_ns()-t)
  times[b]=dict(neural_ns=float(np.median(ts)),compiled_ns=float(np.median(cs)))
 return dict(seed=seed,n=len(g),same=int((ref==out).sum()),correct=int(exact),ambiguous=int(amb.sum()),ask_correct=int((amb&ask).sum()),unsafe=int((amb&~ask).sum()),extra=int((~amb&ask).sum()),default_wrong=default_wrong,prep_ns=prep,times=times)
if __name__=='__main__':
 rows=[acquire(s,a,e,b) for s in list(range(500,532))+list(range(700,732)) for a in [False,True] for e in [0,.1] for b in [1,3,7]];comp=[compile_test(s) for s in [11,29,47]]
 (P/'followup.json').write_text(json.dumps(dict(acquisition=rows,compiled=comp),indent=2));
 for a in [False,True]:
  for e in [0,.1]:
   for b in [1,3,7]:
    z=[x for x in rows if x['active']==a and x['noise']==e and x['budget']==b];print(a,e,b,{k:sum(x[k] for x in z) for k in ['n','correct','admit','admit_correct','observations']})
 print(comp)
