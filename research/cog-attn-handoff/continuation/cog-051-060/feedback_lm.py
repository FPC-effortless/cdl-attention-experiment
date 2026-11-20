"""Train a causal Transformer to predict both instruction words and a grounded effect token."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from panel import Panel,probe
from grounding import NAMES,TRAIN,HELD
R=Path(__file__).resolve().parent/'results'
def run(seed):
    rng=np.random.default_rng(seed+520000)
    torch.manual_seed(seed);env=Panel(rng);model=probe(env,rng,True)
    vocab={w:i+2 for i,w in enumerate(sorted(set(' '.join(TRAIN).replace('{p}','').replace('{q}','').split())|set(NAMES)))}
    effect_token=max(vocab.values())+1
    def data(n,mode):
        inputs=[];labels=[]
        while len(inputs)<n:
            a,b=map(int,rng.choice(5,2,replace=False))
            if mode=='train' and (a-b)%5==1:continue
            if mode!='train':b=(a-1)%5
            phrase=rng.choice(TRAIN) if mode in ('train','reverse') else HELD[mode=='implicit']
            words=phrase.format(p=NAMES[a],q=NAMES[b]).split()
            s=int(rng.integers(32));effect=model.label(env.frame(s),env.frame(env.step(s,a)))
            if effect is None:continue
            ids=[1]+[vocab.get(w,1) for w in words]+[effect_token+effect]
            inputs.append(ids+[0]*(16-len(ids)));labels.append(a)
        return torch.tensor(inputs),torch.tensor(labels)
    train,_=data(1000,'train')
    held={k:data(250,k) for k in ('reverse','joint','implicit')}
    class LanguageModel(nn.Module):
        def __init__(self):
            super().__init__();self.emb=nn.Embedding(effect_token+5,32);self.pos=nn.Embedding(16,32)
            self.core=nn.TransformerEncoderLayer(32,4,64,dropout=0,batch_first=True)
            self.head=nn.Linear(32,effect_token+5)
        def forward(self,x):
            h=self.emb(x)+self.pos(torch.arange(x.size(1)))
            return self.head(self.core(h,src_mask=torch.ones(x.size(1),x.size(1),dtype=torch.bool).triu(1)))
    out={}
    for name in ('all_tokens','effect_only'):
        torch.manual_seed(seed);net=LanguageModel();opt=torch.optim.AdamW(net.parameters(),lr=.003)
        gen=np.random.default_rng(seed)
        for _ in range(300):
            rows=train[gen.integers(len(train),size=48)]
            x,y=rows[:,:-1],rows[:,1:]
            logits=net(x);mask=y!=0 if name=='all_tokens' else y>=effect_token
            loss=nn.functional.cross_entropy(logits[mask],y[mask])
            opt.zero_grad();loss.backward();opt.step()
        out[name]={}
        with torch.no_grad():
            for k,(rows,y) in held.items():
                logits=net.eval()(rows[:,:-1]);ix=(rows!=0).sum(1)-2
                answer=logits[torch.arange(len(rows)),ix,effect_token:].argmax(1)
                out[name][k]=int((answer==y).sum())
    return out
if __name__=='__main__':
    torch.set_num_threads(1);(R/'feedback-lm.json').write_text(json.dumps({s:run(s) for s in (5101,5107,5113)}))
