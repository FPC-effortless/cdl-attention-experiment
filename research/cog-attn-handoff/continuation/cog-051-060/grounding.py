"""Learn the referent from a physical consequence; compare encoders on identical experience."""
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from panel import N, Panel, probe

TRAIN = ("change {p} and keep {q}", "keep {q} and change {p}", "toggle {p} leaving {q} alone")
HELD = ("please keep {q} still then change {p}", "{q} stays as it is so {p} should look different")
NAMES = "dax wug fep blick zorp".split()
ALIASES = "mip lom tive nash vekk".split()


def role_features(words, name):
    i=words.index(name)
    def token(j):
        if not 0<=j<len(words): return "<edge>"
        return "<referent>" if words[j] in NAMES else words[j]
    return {**{f"{j}:{token(i+j)}":1 for j in (-3,-2,-1,1,2,3)}, "relative":i/max(1,len(words)-1)}


class Sequence(nn.Module):
    def __init__(self,vocab,kind):
        super().__init__()
        self.emb=nn.Embedding(vocab,32,padding_idx=0)
        self.pos=nn.Embedding(20,32)
        self.kind=kind
        self.block=(nn.GRU(32,32,batch_first=True) if kind=="gru" else
                    nn.TransformerEncoderLayer(32,4,64,dropout=0,batch_first=True))
        self.head=nn.Linear(32,N)

    def forward(self,x):
        e=self.emb(x)+self.pos(torch.arange(x.shape[1]))
        if self.kind=="gru":
            h=self.block(e)[0]
        else:
            h=self.block(e,src_key_padding_mask=x==0)
        return self.head(h[torch.arange(len(x)),(x!=0).sum(1)-1])


def run(seed,out,reader):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    env=Panel(rng)
    model=probe(env,rng,True)
    names=list(rng.permutation(NAMES))
    aliases=list(rng.permutation(ALIASES))
    def effect(motor):
        while True:
            state=int(rng.integers(32))
            pre,post=env.frame(state),env.frame(env.step(state,motor))
            label=model.label(pre,post)
            if label is not None: return label
    dictionary={}
    for a in range(N):
        for word in [names[a],aliases[a]]:
            dictionary[word]=effect(a)
    canonical={w:NAMES[a] for w,a in dictionary.items()}
    def example(p,q,form,novel=False):
        utterance=form.format(p=(aliases if novel else names)[p],q=names[q])
        return [canonical.get(w,w) for w in utterance.split()],effect(p),utterance
    training=[]
    for _ in range(1200):
        p,q=map(int,rng.choice(N,2,replace=False))
        if (p-q)%N!=1:
            training.append(example(p,q,rng.choice(TRAIN)))
    held={}
    for slice_name in ('familiar','reversed_pair','new_syntax','implicit','joint','unsupported_alias'):
        block=[]
        for _ in range(160):
            p,q=map(int,rng.choice(N,2,replace=False))
            if slice_name=='familiar' and (p-q)%N==1: q=(p+1)%N
            if slice_name in ('reversed_pair','joint'): q=(p-1)%N
            form=rng.choice(TRAIN) if slice_name in ('familiar','reversed_pair','unsupported_alias') else HELD[int(slice_name=='implicit')]
            x,y,text=example(p,q,form,slice_name=='joint')
            if slice_name=='unsupported_alias': x[x.index(NAMES[p])]="unseen"
            block.append((x,y,text))
        held[slice_name]=block
    samples=training+sum(held.values(),[])
    # New grammar tokens remain unknown until observed during training.
    vocab={w:i+2 for i,w in enumerate(sorted({w for x,_,_ in training for w in x}))}
    inputs=np.zeros((len(samples),16),int)
    for i,(x,_,_) in enumerate(samples):
        inputs[i,:len(x)]=[vocab.get(w,1) for w in x]
    truth=np.array([y for _,y,_ in samples]); n=len(training)
    features=[]; labels=[]
    for x,y,_ in training:
        for word in x:
            if word in NAMES:
                features.append(role_features(x,word)); labels.append(int(word==NAMES[y]))
    vec=DictVectorizer(); clf=LogisticRegression(C=3).fit(vec.fit_transform(features),labels)
    def choose(x):
        candidates=[w for w in x if w in NAMES]
        z=clf.decision_function(vec.transform([role_features(x,w) for w in candidates]))
        return NAMES.index(candidates[np.argmax(z)])
    preds={"roles":np.array([choose(x) for x,_,_ in samples])}
    x=torch.from_numpy(inputs); y=torch.from_numpy(truth)
    for kind in ('gru','transformer'):
        net=Sequence(max(vocab.values())+1,kind)
        optim=torch.optim.AdamW(net.parameters(),lr=.003)
        for step in range(300):
            idx=rng.integers(n,size=48)
            loss=nn.functional.cross_entropy(net(x[idx]),y[idx])
            optim.zero_grad(); loss.backward(); optim.step()
        with torch.no_grad(): preds[kind]=net.eval()(x).argmax(1).numpy()
        torch.save(net.state_dict(),out/f'{kind}.pt')
    # Use the upstream CSM reader and optimization routine with this learned motor vocabulary.
    orig=reader.V.copy()
    reader.V.clear(); reader.V.update(vocab); reader.V['<unknown>']=1
    net=reader.Reader(seed,True)
    docs=np.zeros((8,2,16),int)
    for j,name in enumerate(NAMES): docs[j,:,0]=vocab[name]
    for step in range(1,351):
        idx=rng.integers(n,size=(20,2))
        net.step((np.tile(docs,(len(idx),1,1,1)).reshape(len(idx),16,16),inputs[idx],truth[idx],None),step)
    memory=net.forward(docs)[0].sum(1)
    preds["csm"]=np.argmax(net.forward(inputs)[0]@memory[:N].T,axis=1)
    np.savez(out/'csm.npz',E=net.E,a=net.a,b=net.b)
    reader.V.clear(); reader.V.update(orig)
    # Select a previously observed utterance for the requested motor; evaluate its physical effect.
    utterances={y:(text,y) for _,y,text in training}
    joint=held['joint']; diagnostic=[]
    results={}
    for key,items in held.items():
        ids=slice(n,n+len(items)); n+=len(items)
        results[key]={k:int((p[ids]==truth[ids]).sum()) for k,p in preds.items()}
        if key=='joint':
            diagnostic=[dict(text=text,true=int(label),action=int(action),reply=utterances[int(action)][0])
                        for (_,label,text),action in zip(items,preds['roles'][ids])]
    import joblib
    joblib.dump(dict(vec=vec,clf=clf,dictionary=dictionary,vocab=vocab,utterances=utterances),out/'roles.joblib')
    return dict(names=names,aliases=aliases,trials_per_slice=160,training=len(training),accuracy_counts=results,joint=diagnostic)
