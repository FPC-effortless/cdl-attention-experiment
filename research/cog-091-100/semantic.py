"""COG-ATTN-092-095 bounded, trainable char-grounding pilot.

This is a synthetic router benchmark with a 7-class supervised action inventory,
NOT an integrated COG-ATTN agent. It uses no self-attention or QK^T weights.
"""
from __future__ import annotations
import argparse, hashlib, json, random, re
from time import perf_counter_ns
from collections import Counter
import torch
from torch import nn

ACTIONS=('schedule','cancel','move','assign','approve','reject')
TRAIN_FORMS=(('schedule','scheduling'),('cancel','cancelling'),('move','moving'),
             ('assign','assigning'),('approve','approving'),('reject','rejecting'))
HELD_FORMS=('scheduled','cancelled','moved','assigned','approved','rejected')
TEMPLATES=(('please {a} actually {b}',1),
           ('do not {a} choose {b} instead',1),
           ('please {a} rather than {b}',0),
           ('use {a} not {b}',0),
           ('perhaps {a} or {b}',6))
SEEDS=(9107,9203,9301)
DIM=256
MAXLEN=12


def charvec(word):
    padded='^'+word+'$'
    vec=[0.0]*DIM
    for width in (2,3):
        for i in range(len(padded)-width+1):
            part=padded[i:i+width]
            h=int.from_bytes(hashlib.sha256(part.encode()).digest()[:4],'big')%DIM
            vec[h]+=1.0
    norm=sum(z*z for z in vec)**0.5 or 1.0
    return [z/norm for z in vec]


def vectorize(text, cache):
    words=re.findall('[a-z]+',text.lower())[:MAXLEN]
    features=[cache.setdefault(w,charvec(w)) for w in words]
    features += [[0.0]*DIM]*(MAXLEN-len(features))
    return features,len(words)


def make_data(seed):
    rng=random.Random(seed)
    pairs=[(i,j) for i in range(6) for j in range(6) if i!=j]
    rng.shuffle(pairs)
    # Split by entire ordered action pair to test compositional transfer.
    train_pairs=set(pairs[:20]);dev_pairs=set(pairs[20:24]);test_pairs=set(pairs[24:])
    assert not(train_pairs & dev_pairs or train_pairs & test_pairs or dev_pairs & test_pairs)
    def examples(split):
        take=train_pairs if split=='train' else dev_pairs if split=='dev' else test_pairs
        rows=[]
        for ai,bi in sorted(take):
            for temp,choice in TEMPLATES:
                for variation in (0,1):
                    a=TRAIN_FORMS[ai][variation]
                    b=TRAIN_FORMS[bi][1-variation]
                    if split=='test' and variation==1:
                        a,b=HELD_FORMS[ai],HELD_FORMS[bi]
                    msg=temp.format(a=a,b=b)
                    answer=(ai if choice==0 else bi if choice==1 else 6)
                    rows.append(dict(text=msg,answer=answer,template=temp,pair=(ai,bi),
                                     novelty='unseen_inflection' if split=='test' and variation==1 else 'familiar'))
        return rows
    out={split:examples(split) for split in ('train','dev','test')}
    # Disjoint pair splits are insufficient if any sentence ignores its other pair member.
    phrases=[{r['text'] for r in out[s]} for s in ('train','dev','test')]
    assert not (phrases[0]&phrases[1] or phrases[0]&phrases[2] or phrases[1]&phrases[2])
    return out


def to_tensor(rows,cache):
    data=[vectorize(r['text'],cache) for r in rows]
    return torch.tensor([x for x,_ in data],dtype=torch.float32),torch.tensor([z for _,z in data],dtype=torch.long),torch.tensor([r['answer'] for r in rows],dtype=torch.long)


class Router(nn.Module):
    def __init__(self,mode,freeze_encoder=False):
        super().__init__();self.mode=mode
        self.encoder=nn.Linear(DIM,32,bias=False)
        if freeze_encoder:self.encoder.weight.requires_grad_(False)
        if mode=='bag':self.out=nn.Sequential(nn.Linear(32,48),nn.ReLU(),nn.Linear(48,7))
        elif mode=='recurrent':
            self.gru=nn.GRU(32,48,batch_first=True)
            self.out=nn.Linear(48,7)
        elif mode=='self_attention':
            # Matched supervised-data transformer baseline; small model, not pretrained.
            self.positions=nn.Embedding(MAXLEN,32)
            layer=nn.TransformerEncoderLayer(d_model=32,nhead=4,dim_feedforward=64,dropout=0.0,batch_first=True)
            self.transformer=nn.TransformerEncoder(layer,num_layers=2,enable_nested_tensor=False)
            self.out=nn.Linear(32,7)
        elif mode=='parallel_specialists':
            # Three local temporal specialists processed with independent convolutions.
            self.specialists=nn.ModuleList([nn.Conv1d(32,24,k,padding=k//2) for k in (1,3,5)])
            self.out=nn.Sequential(nn.Linear(72,48),nn.ReLU(),nn.Linear(48,7))
        else:raise ValueError(mode)

    def forward(self,x,lengths):
        t=torch.relu(self.encoder(x))
        if self.mode=='bag':
            mask=(torch.arange(x.shape[1],device=x.device)[None,:]<lengths[:,None]).float()
            feature=(t*mask[:,:,None]).sum(1)/lengths[:,None].clamp(min=1)
        elif self.mode=='recurrent':
            feature=self.gru(t)[0][torch.arange(x.shape[0]),lengths-1,:]
        elif self.mode=='self_attention':
            positions=torch.arange(x.shape[1],device=x.device)[None,:]
            mask=positions>=lengths[:,None]
            z=self.transformer(t+self.positions(positions),src_key_padding_mask=mask)
            feature=(z*(~mask)[:,:,None]).sum(1)/lengths[:,None].clamp(min=1)
        else:
            channels=t.transpose(1,2)
            pooled=[]
            mask=torch.arange(x.shape[1],device=x.device)[None,None,:]<lengths[:,None,None]
            for conv in self.specialists:
                z=torch.relu(conv(channels))
                pooled.append(z.masked_fill(~mask,-1e4).amax(dim=2))
            feature=torch.cat(pooled,dim=1)
        return self.out(feature)


def lex_first(text):
    words=set(re.findall('[a-z]+',text.lower()))
    for word in re.findall('[a-z]+',text.lower()):
        for i,variants in enumerate(TRAIN_FORMS):
            if word in variants: return i
    return 6


def fit_evaluate(split,seed,mode,freeze=False,epochs=38):
    torch.manual_seed(seed)
    cache={}
    tr=to_tensor(split['train'],cache);dv=to_tensor(split['dev'],cache);te=to_tensor(split['test'],cache)
    model=Router(mode,freeze);optimizer=torch.optim.AdamW(filter(lambda p:p.requires_grad,model.parameters()),lr=0.007,weight_decay=0.0001)
    # Select epoch using ONLY disjoint development pairs, not held test pairs.
    best,chosen=-1.0,None
    begin_train=perf_counter_ns()
    for epoch in range(epochs):
        model.train()
        idx=torch.randperm(len(tr[2]))
        for inds in idx.split(48):
            logits=model(tr[0][inds],tr[1][inds])
            loss=nn.functional.cross_entropy(logits,tr[2][inds])
            optimizer.zero_grad();loss.backward();optimizer.step()
        model.eval()
        with torch.no_grad():
            val=(model(dv[0],dv[1]).argmax(1)==dv[2]).float().mean().item()
        if val>best+1e-8:
            best=val;chosen=(epoch+1,{k:v.detach().clone() for k,v in model.state_dict().items()})
    train_ns=perf_counter_ns()-begin_train
    model.load_state_dict(chosen[1]);model.eval()
    with torch.no_grad():
        dev_probs=torch.softmax(model(dv[0],dv[1]),dim=1)
        # Choose a commit threshold exclusively on development action precision.
        dev_best=[]
        for threshold in (i/100 for i in range(101)):
            selected=(dev_probs.max(1).values>=threshold)&(dev_probs.argmax(1)!=6)
            attempted=int(selected.sum())
            valid=int(((dev_probs.argmax(1)==dv[2])&selected).sum())
            if attempted==0 or valid/attempted>=0.95:
                dev_best.append((valid,attempted,-threshold,threshold))
        cutoff=max(dev_best)[-1] if dev_best else 1.01
        before=perf_counter_ns();output=model(te[0],te[1]);test_ns=perf_counter_ns()-before
        prob=torch.softmax(output,dim=1);p=prob.argmax(1)
        y=te[2]
        actions=(p!=6)&(prob.max(1).values>=cutoff)
        action_count=int(actions.sum()); action_correct=int(((p==y)&actions).sum())
        online_cache={};begin_online=perf_counter_ns()
        for case in split['test']:
            vec,length=vectorize(case['text'],online_cache)
            inp=torch.tensor([vec],dtype=torch.float32)
            _=model(inp,torch.tensor([length]))
        online_ns=perf_counter_ns()-begin_online
    by_kind={}
    for kind in ('familiar','unseen_inflection'):
        indices=[i for i,r in enumerate(split['test']) if r['novelty']==kind]
        by_kind[kind]=dict(n=len(indices),correct=sum(int(p[i]==y[i]) for i in indices))
    ambiguous=[i for i,r in enumerate(split['test']) if r['answer']==6]
    ambiguous_pair_top2=sum(int(set(split['test'][i]['pair'])==set(prob[i,:6].topk(2).indices.tolist())) for i in ambiguous)
    errors=[dict(text=split['test'][i]['text'],target=int(y[i]),predicted=int(p[i]),
                 confidence=round(float(prob[i].max()),4)) for i in range(len(y)) if p[i]!=y[i]]
    return dict(mode=mode,freeze_encoder=freeze,seed=seed,params=sum(p.numel() for p in model.parameters()),
                dev_best=best,selected_epoch=chosen[0],n=len(y),correct=int((p==y).sum()),
                by_kind=by_kind,train_ms=round(train_ns/1e6,3),
                test_batch_ms=round(test_ns/1e6,3),
                online_all_72_ms=round(online_ns/1e6,3),
                gated_threshold=round(cutoff,3),gated_attempted=action_count,
                gated_correct=action_correct,gated_wrong=action_count-action_correct,
                ambiguous_n=len(ambiguous),ambiguous_pair_top2=ambiguous_pair_top2,
                errors=errors)


def run():
    torch.set_num_threads(1)
    rows=[];controls=[]
    for seed in SEEDS:
        split=make_data(seed)
        t=split['test']
        controls.append(dict(seed=seed,n=len(t),lexical_first=sum(lex_first(r['text'])==r['answer'] for r in t),
                             train_n=len(split['train']),dev_n=len(split['dev']),test_pairs=len({tuple(r['pair']) for r in t})))
        for mode,freeze in [('bag',False),('recurrent',True),('recurrent',False),('parallel_specialists',False),('self_attention',False)]:
            r=fit_evaluate(split,seed,mode,freeze)
            rows.append(r)
            print(seed,mode,freeze,r['correct'],'/',r['n'],'dev',round(r['dev_best'],3),'train_ms',r['train_ms'])
    return dict(name='COG-ATTN-092–095 synthetic grounded-language pilot',seeds=list(SEEDS),
                controls=controls,models=rows,
                limitations=['Six predefined action classes and one defer class; does not learn arbitrary actions',
                             'Training labels generated by template grammar, not by natural language feedback',
                             'All templates seen during training, but ordered action pairs disjoint',
                             'Test synonyms share stems with trained forms; not broad zero-shot language',
                             'Fitted component is not integrated into full COG workflow, verifier or ledger',
                             'Online query timing includes per-query character encoding and forward pass, but not end-to-end COG verification or writes',
                             'Tiny trained-from-scratch Transformer is a diagnostic baseline, not a pretrained frontier model'])

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='semantic-results.json');args=ap.parse_args()
    result=run()
    with open(args.out,'w') as f:json.dump(result,f,indent=2,sort_keys=True)
