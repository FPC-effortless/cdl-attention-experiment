"""No simulator/evaluator import. Learner receives observations and controls only."""
import numpy as np
LAG=3
ARMS=('predictive-adaptive','predictive-frozen','pca-adaptive','narx-adaptive','persistence')
def read_view(path):
 import json
 return {r['id']:(np.array(r['features']['o']),np.array(r['features']['u'])) for r in json.loads(path.read_text())['rows']}
def ridge(x,y,p):
 return np.linalg.solve(x.T@x + p*np.eye(x.shape[1]),x.T@y)
def windows(o):
 return np.concatenate([o[i:len(o)-LAG+i+1] for i in range(LAG)],axis=1)
def feats(z,u):
 # quadratic NARX without sensor-sensor 48x48 explosion; compact also gets
 # latent interactions. All arms get state/control terms via same feature rule.
 dz=z.shape[-1]
 ix=np.triu_indices(dz); ix=tuple(i[(ix[0]//16 == ix[1]//16)] for i in ix) if dz>16 else ix
 return np.concatenate([np.ones((*z.shape[:-1],1)),z,u,
                         z[...,ix[0]]*z[...,ix[1]],(z[...,None]*u[...,None,:]).reshape(*z.shape[:-1],2*dz),
                         u[...,0:1]*u[...,1:2],u*u],axis=-1)
def fit_basis(x,y,u):
 # past-only x at time t, future o(t+1) target, causal control u(t). Remove
 # the expected linear direct/control nuisance from future targets before SVD.
 yr=y-u@ridge(u,y,1.)
 chol=np.linalg.cholesky(x.T@x+4*np.eye(x.shape[1]))
 xw=np.linalg.solve(chol,x.T).T
 left=np.linalg.svd(xw.T@yr,full_matrices=False)[0][:,:4]
 return np.linalg.solve(chol.T,left)*np.sqrt(len(x))
def fit_arm(data,mode):
 obs=np.concatenate([o for o,u in data.values()]);mu=obs.mean(0);sd=np.maximum(obs.std(0),1e-6)
 episodes=[((o-mu)/sd,u) for o,u in data.values()]
 histories=[windows(o) for o,_ in episodes]
 x=np.concatenate([h[:-1] for h in histories]); nxt=np.concatenate([h[1:] for h in histories]);u=np.concatenate([u[LAG-1:] for _,u in episodes])
 if mode=='predictive':w=fit_basis(x,nxt[:,-16:],u)
 elif mode=='pca':w=np.linalg.svd(x,full_matrices=False)[2][:4].T
 else:w=np.eye(x.shape[1])
 z=x@w
 # Scale state for matched numerical regularization; zero variance guarded.
 w=w/np.maximum(z.std(0),1e-6);z=x@w
 design=feats(z,u);w_dyn=ridge(design,nxt[:,-16:],2.)
 return dict(w=w,mean=mu,std=sd,dynamics=w_dyn,gram_cond=np.linalg.cond(design.T@design+2*np.eye(design.shape[1])))
def encode(m,o):
 return ((o[-LAG:]-m['mean'])/m['std']).ravel()@m['w']
def prepare(m,o,u,mode):
 h=windows((o-m['mean'])/m['std']); m0=m['dynamics']
 if mode.endswith('adaptive'):
  # Fixed common support and regularization for all adaptive arms.
  z=h[:-1][-16:]@m['w'];d=feats(z,u[LAG-1:][-16:]);y=h[1:][-16:,-16:];e=y-d@m0
  m0=m0+d.T@np.linalg.solve(d@d.T+10*np.eye(len(d)),e)
 return ((o[-LAG:]-m['mean'])/m['std']).copy(),m0
def rollout(m,history,beta,plans,mode):
 if mode=='persistence':
  return np.broadcast_to(history[-1]*m['std']+m['mean'],(len(plans),len(plans[0]),len(m['mean']))).copy()
 h=np.broadcast_to(history,(len(plans),*history.shape)).copy();pred=[]
 for t in range(plans.shape[1]):
  v=feats(h.reshape(len(h),-1)@m['w'],plans[:,t])@beta
  v=np.clip(v,-8,8)
  pred.append(v*m['std']+m['mean'])
  h=np.concatenate([h[:,1:],v[:,None]],axis=1)
 return np.stack(pred,axis=1)
