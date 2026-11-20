"""Equal-input action classifiers trained from scratch with the same update budget."""
from collections import Counter
import json
import re
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.metrics import average_precision_score

P=Path('/workspace/cog-next-private/authentic')
sys.path.insert(0,str(P))
from evaluation import measure_admissions, summarize, threshold_selection, eligible
from models import history,cues,fit,txt_features,prob
from scipy.sparse import csr_matrix,hstack

class Model(nn.Module):
    def __init__(self,n,kind):
        super().__init__()
        self.kind=kind
        self.emb=nn.Embedding(n,24,padding_idx=0)
        self.pos=nn.Embedding(64,24)
        self.decay=nn.Parameter(torch.zeros(24))
        if kind=='gru':self.core=nn.GRU(24,24,batch_first=True)
        if kind=='transformer':self.core=nn.TransformerEncoderLayer(24,3,48,dropout=0,batch_first=True)
        self.head=nn.Linear(32,1)
    def forward(self,x,c):
        h=self.emb(x)+self.pos(torch.arange(x.shape[1]))
        if self.kind=='transformer':
            h=self.core(h,src_key_padding_mask=x==0)
        elif self.kind=='gru':
            h=self.core(h)[0]
        else:
            # Stable diagonal linear state-space recurrence with an input-dependent readout.
            alpha=self.decay.sigmoid()
            seq=[]; state=torch.zeros(len(x),24)
            for inp in h.unbind(1):
                state=alpha*state+(1-alpha)*inp
                seq.append(state)
            h=torch.stack(seq,1)
        last=(x!=0).sum(1).clamp(min=1)-1
        return self.head(torch.cat([h[torch.arange(len(x)),last],c],1)).flatten()

def main():
    rows=json.loads((P/'combined.json').read_text())
    target=np.array([r['target'] for r in rows]); group=np.array([r['issue'] for r in rows])
    tr=np.array([i for i,r in enumerate(rows) if r['split']=='train' and (r['strict'] or (r['target'] and r['delta_seconds']==0))])
    splits={s:np.array([i for i,r in enumerate(rows) if r['split']==s and r['strict']]) for s in ('validation','test','external')}
    va=splits['validation']
    text=[re.findall(r"\b\w+\b",r['text'])[:64] for r in rows]
    vocab={w:j+2 for j,(w,_) in enumerate(Counter(t for i in tr for t in text[i]).most_common(6000))}
    x=torch.zeros(len(rows),64,dtype=torch.long)
    for i,words in enumerate(text):
        ids=[vocab.get(w,1) for w in words] or [1]; x[i,:len(ids)]=torch.tensor(ids)
    ctx=history(rows);c=torch.tensor(ctx,dtype=torch.float);y=torch.tensor(target,dtype=torch.float)
    cue=np.array([cues(r) for r in rows]); dummy=np.zeros((len(rows),3))
    xx,vec=txt_features(np.array([r['text'][:12000] for r in rows],object),tr)
    feat=hstack([xx,csr_matrix(ctx),csr_matrix(cue)],format='csr')
    base=fit(feat,target,tr,va); linear=prob(base,feat)
    out={}
    for seed in (67003,67009,67021):
        modes={'attention_free_linear':linear}
        for kind in ('transformer','gru','diagonal_ssm'):
            torch.manual_seed(seed);rng=np.random.default_rng(seed);m=Model(max(vocab.values())+1,kind)
            opt=torch.optim.AdamW(m.parameters(),lr=.002)
            for step in range(180):
                ids=rng.choice(tr,32)
                loss=nn.functional.binary_cross_entropy_with_logits(m(x[ids],c[ids]),y[ids])
                opt.zero_grad();loss.backward();opt.step()
            m.eval()
            with torch.no_grad(): modes[kind]=torch.cat([m(a,b).sigmoid() for a,b in zip(x.split(64),c.split(64))]).numpy()
            torch.save(m.state_dict(),P/f'{kind}-{seed}.pt')
        out[seed]={}
        for key,p in modes.items():
            t,s=threshold_selection(p,cue,dummy,target,va,False)
            sel=eligible(p,cue,dummy,t,s)
            out[seed][key]={split:{**summarize(target[idx],p[idx]),**measure_admissions(target[idx],sel[idx],group[idx])} for split,idx in splits.items()}
    path=Path(__file__).resolve().parent/'results/matched.json'
    path.write_text(json.dumps(out))
if __name__=='__main__':
    torch.set_num_threads(1)
    main()
