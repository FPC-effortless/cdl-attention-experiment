"""Supervised sparse language-to-record routing with paired, episodic references."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import hashlib,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parent
WORDS='amber gold navy blue olive green ruby red chalk white coal black lilac violet coral pink'.split()
LEX='file for entry choose over ignore and rather than'.split()+WORDS
V={w:i+1 for i,w in enumerate(LEX)}
def encode(text):
    ids=[V[w] for w in text.split()]
    return ids+[0]*(6-len(ids))
def data(seed,batch,mode):
    rng=np.random.default_rng(seed)
    xs=[];ys=[];targets=[];values=[]
    for _ in range(batch):
        order=rng.permutation(8)
        def name(c):return WORDS[2*c+rng.integers(2)]
        xs.append([encode(rng.choice(['file for ','entry '])+name(c)) for c in order for j in range(2)])
        a,b=rng.choice(8,2,replace=False)
        p,q=name(order[a]),name(order[b])
        form=rng.integers(3)
        def query(p,q):
            if mode=='held':return f'over {q} choose {p}'
            return [f'choose {p} over {q}',f'ignore {q} and choose {p}',f'{p} rather than {q}'][form]
        ys.append([encode(query(p,q)),encode(query(q,p))])
        targets.append([a,b]);values.append(rng.permutation(8))
    return [np.array(t) for t in (xs,ys,targets,values)]
class Reader:
    def __init__(self,seed,order):
        r=np.random.default_rng(seed);self.order=order
        self.E=r.normal(0,.25,(len(V)+1,32));self.E[0]=0
        self.a=r.normal(.65,.1,32);self.b=np.full(32,.3)
        self.par=[self.E,self.a,self.b];self.m=[p*0 for p in self.par];self.v=[p*0 for p in self.par]
    def forward(self,words):
        x=words.reshape(-1,words.shape[-1]);h=np.zeros((len(x),32));hs=[h]
        for t in x.T:
            z=self.E[t]+self.a*h+self.b if self.order else h+self.E[t]*self.a+self.b
            h=np.where(t[:,None]!=0,np.tanh(z) if self.order else z,h);hs.append(h)
        idx=np.argpartition(h,-8,axis=1)[:,-8:];mask=np.zeros_like(h);np.put_along_axis(mask,idx,1,axis=1)
        f=mask*np.maximum(h,0);norm=np.sqrt((f*f).sum(1,keepdims=True)+1e-6);f=f/norm
        return f.reshape(*words.shape[:-1],32),(x,hs,mask*(h>0),norm,f)
    def backward(self,cache,grad):
        x,hs,mask,norm,f=cache
        g=grad.reshape(f.shape);g=mask*(g-f*(g*f).sum(1,keepdims=True))/norm
        ge,ga,gb=[p*0 for p in self.par]
        for j in reversed(range(x.shape[1])):
            t=x[:,j];active=(t!=0)[:,None]
            z=g*active*((1-hs[j+1]**2) if self.order else 1)
            np.add.at(ge,t,z if self.order else z*self.a)
            ga+=(z*(hs[j] if self.order else self.E[t])).sum(0);gb+=z.sum(0)
            g=(g*(~active)+z*self.a) if self.order else g
        return ge,ga,gb
    def step(self,ds,step):
        x,y,target,_=ds
        f,c=self.forward(x);q,d=self.forward(y);s=f.reshape(len(x),8,2,32).sum(2)
        logits=6*q@s.transpose(0,2,1);p=np.exp(logits-logits.max(2,keepdims=True));p/=p.sum(2,keepdims=True)
        np.put_along_axis(p,target[:,:,None],np.take_along_axis(p,target[:,:,None],2)-1,axis=2)
        dz=6*p/target.size
        gq=dz@s;gs=dz.transpose(0,2,1)@q
        g1=self.backward(c,np.repeat(gs,2,axis=1));g2=self.backward(d,gq)
        for w,m,v,i,j in zip(self.par,self.m,self.v,g1,g2):
            g=i+j;m*=.9;m+=.1*g;v*=.999;v+=.001*g*g
            w-=.008*(m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1e-8)
def scores(model,ds):
    x,y,t,values=ds;f=model.forward(x)[0].reshape(len(x),8,2,32);q=model.forward(y)[0]
    h=q@f.sum(2).transpose(0,2,1);over=np.einsum('bqd,bskd->bqsk',q,f)
    assert np.allclose(h,over.sum(-1),atol=1e-12)
    out={}
    for name,z in [('sum',h),('nearest',over.max(-1))]:
        win=z.argmax(2);tied=np.isclose(z,z.max(2,keepdims=True),rtol=0,atol=1e-10).sum(2)>1
        win[tied]=-1
        ok=win==t
        out[name]={'route':int(ok.sum()),'both':int(ok.all(1).sum()),'abstain':int(tied.sum()),'n':int(ok.size)}
    return out
def main():
    config=dict(seeds=[11,23,37],steps=400,episodes_per_step=16,evaluation_episodes=512,
                slots=8,supports_per_slot=2,gate='95% individual and both swap directions on each cohort/seed; ties abstain',generator='synthetic single author; three training forms, one held form')
    panels={m:data(i,512,m) for m,i in [('familiar',2801001),('held',2801002)]}
    np.savez_compressed(ROOT/'panels.npz',**{f'{m}{j}':a for m,ds in panels.items() for j,a in enumerate(ds)})
    config['panel_digest']=hashlib.sha256((ROOT/'panels.npz').read_bytes()).hexdigest()
    (ROOT/'design.json').write_text(json.dumps(config,indent=2));rows=[]
    for seed in config['seeds']:
        for order in (False,True):
            model=Reader(seed,order);start=time.perf_counter()
            for step in range(1,401):model.step(data(seed*10000+step,16,'train'),step)
            name='recurrent' if order else 'pooled'
            np.savez(ROOT/f'{name}-{seed}.npz',E=model.E,a=model.a,b=model.b)
            rows.append(dict(seed=seed,reader=name,seconds=time.perf_counter()-start,**{m:scores(model,d) for m,d in panels.items()}))
            (ROOT/'results.json').write_text(json.dumps(rows,indent=2));print(name,seed,flush=True)
if __name__=='__main__':main()
