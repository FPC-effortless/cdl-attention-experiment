"""Learn role scores from state transitions and qualify predicted commands."""
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from binding import KNOWN, Roles, lexical_ids, words_for_seed
from language import NAMES, encode
from paraphrases import fixture

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
MODELS = ("pretrained", "random", "local")
ROOT = Path(__file__).resolve().parent
ACTIONS = torch.tensor([(op,s,d) for op in range(4) for s in range(3) for d in range(3) if s!=d])


def texts(raw,verb_words):
    sentences,masks = [],[]
    for tokens,entities,*_ in raw:
        parts = ["adjust" if t in verb_words else t for t in tokens]
        spans = [[] for _ in entities]
        at = 0
        for token,part in zip(tokens,parts):
            if token in entities:
                spans[entities.index(token)].append((at,at+len(part)))
            at += len(part)+1
        sentences.append(" ".join(parts))
        masks.append(spans)
    return sentences,masks


def load_encoder(mode):
    folder = ROOT/"minilm"
    tokenizer = AutoTokenizer.from_pretrained(folder,local_files_only=True,use_fast=True)
    model = (AutoModel.from_pretrained(folder,local_files_only=True,use_safetensors=True) if mode=="pretrained"
             else AutoModel.from_config(AutoConfig.from_pretrained(folder,local_files_only=True)))
    return tokenizer,model.eval()


def contextual(raw,verb_words,teacher):
    tokenizer,model = teacher
    sentences,spans = texts(raw,verb_words)
    features = []
    with torch.no_grad():
        for k in range(0,len(sentences),96):
            batch = tokenizer(sentences[k:k+96],padding=True,return_offsets_mapping=True,return_tensors="pt")
            offsets = batch.pop("offset_mapping")
            hidden = model(**batch).last_hidden_state
            masks = torch.zeros((len(hidden),3,hidden.shape[1]),dtype=torch.bool)
            for row,entities in enumerate(spans[k:k+96]):
                start,end = offsets[row].T
                for ent,locations in enumerate(entities):
                    for a,b in locations:
                        masks[row,ent] |= (end>a)&(start<b)&(end>start)
            assert masks.any(2).all()
            weights = masks.float()/masks.sum(-1,keepdim=True)
            mention = torch.bmm(weights,hidden)
            features.append(torch.cat((hidden[:,:1],mention),dim=1))
    return torch.cat(features)


class ContextHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.roles = nn.Sequential(nn.LayerNorm(768),nn.Linear(768,96),nn.GELU(),nn.Dropout(.15),nn.Linear(96,2))
        self.clarity = nn.Sequential(nn.LayerNorm(384),nn.Linear(384,64),nn.GELU(),nn.Dropout(.2),nn.Linear(64,1))

    def role(self,x):
        mentions = x[:,1:]
        context = x[:,:1].expand(-1,3,-1)
        scores = self.roles(torch.cat((mentions,context),dim=-1))
        return scores[:,ACTIONS[:6,1],0]+scores[:,ACTIONS[:6,2],1]

    def clear(self,x):
        return self.clarity(x[:,0]).squeeze(-1)


class LocalHeads(nn.Module):
    def __init__(self,vocab,entities,verbs):
        super().__init__()
        self.role_net = Roles(vocab,entities,verbs,True)
        self.clear_net = nn.Embedding(vocab,1,padding_idx=0)

    def role(self,x):
        return self.role_net(x,ACTIONS[:6])

    def clear(self,x):
        return self.clear_net(x).squeeze(-1).sum(-1)


def grounding_and_gate_fit(model,x,allowed,ambiguous,seed):
    optimizer = torch.optim.AdamW(model.parameters(),lr=.002)
    rng = torch.Generator().manual_seed(seed+1441)
    role_ok = allowed.reshape(-1,4,6).float().mean(1)
    model.train()
    for _ in range(600):
        good = torch.randint(len(x),(48,),generator=rng)
        bad = torch.randint(len(ambiguous),(24,),generator=rng)
        logits = model.role(x[good])
        likelihood = (logits.softmax(-1)*role_ok[good]).sum(1).clamp_min(1e-12)
        clarity = torch.cat((model.clear(x[good]),model.clear(ambiguous[bad])))
        labels = torch.cat((torch.ones(len(good)),torch.zeros(len(bad))))
        loss = -likelihood.log().mean()+nn.functional.binary_cross_entropy_with_logits(clarity,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()


def infer_word_probs(model,x,allowed,ids,word_ids):
    with torch.no_grad():
        roles = model.role(x).softmax(-1)
        evidence = (roles[:,None,:]*allowed.reshape(-1,4,6)).sum(2)
        z = (.98*evidence+.02/512).log()
        return {i:z[ids==i].sum(0).softmax(-1) for i in word_ids}


def predict(model,x,ids,lexicon,compatible,gold,has_label):
    with torch.no_grad():
        role = torch.cat([model.role(batch).softmax(-1) for batch in x.split(128)])
        clear = torch.cat([model.clear(batch).sigmoid() for batch in x.split(128)])
    op = torch.stack([lexicon[int(i)] for i in ids])
    total = (op[:,:,None]*role[:,None,:]).flatten(1)
    confidence,pred = total.max(-1)
    return dict(pred=pred.tolist(),confidence=confidence.tolist(),clear=clear.tolist(),
                correct=(pred==gold).tolist() if has_label else None,
                effect=compatible[torch.arange(len(pred)),pred].tolist() if has_label else None)


def gate(row,action,clarity):
    return (np.array(row["confidence"])>=action)&(np.array(row["clear"])>=clarity)


def calibrate(clear,ambiguous):
    best=(-1,0,0)
    thresholds=[0.,.5,.8,.9,.95,.98,.99,.995,.999,.9999,1.01]
    for a in thresholds:
        for c in thresholds:
            good = gate(clear,a,c)
            bad = gate(ambiguous,a,c)
            count = int(good.sum())
            wrong = int((good&~np.array(clear["correct"])).sum())
            if wrong<=.01*count and bad.sum()<=.01*len(bad):
                best = max(best,(count,a,c))
    return dict(cal_coverage=best[0],action=best[1],clarity=best[2])


def run(seed,arm):
    extra = words_for_seed(seed)
    verbs = KNOWN+extra
    train,support,ambig,cal,test = fixture(seed,extra)
    vocabulary = {t:i+4 for i,t in enumerate(sorted({w for r in train+support+ambig for w in r[0] if w not in NAMES}))}
    entities = list(range(len(vocabulary)+4,len(vocabulary)+7))
    word_ids = [vocabulary[w] for w in verbs]
    modes = ("pretrained",) if arm=="candidate" else ("random","local")
    results = []
    for mode in modes:
        torch.manual_seed(seed)
        teacher = load_encoder(mode) if mode!="local" else None
        model = ContextHead() if teacher else LocalHeads(len(vocabulary)+7,entities,word_ids)

        def prepare(raw):
            x,compatible,gold,_ = encode(raw,vocabulary,entities)
            if teacher:
                x = contextual(raw,verbs,teacher)
            ids = lexical_ids(raw,vocabulary,verbs)
            return (x,ids,compatible,gold)

        x,ids,allowed,_ = prepare(train)
        x_bad,*_ = prepare(ambig)
        grounding_and_gate_fit(model,x,allowed,x_bad,seed)
        # Derive the verb lexicon from familiar and physically separate support observations.
        s,si,sa,_ = prepare(support)
        lexicon = infer_word_probs(model,x[:256],allowed[:256],ids[:256],word_ids)
        support_lexicon = infer_word_probs(model,s,sa,si,word_ids)
        lexicon.update({vocabulary[w]:support_lexicon[vocabulary[w]] for w in extra})
        heads_path = Path(os.environ["TRAINLAB_OUTPUT_DIR"])/f"{mode}-heads.npz"
        np.savez_compressed(heads_path,**{k:v.detach().numpy() for k,v in model.state_dict().items()})

        def result(raw,label):
            h,t,c,g = prepare(raw)
            return predict(model,h,t,lexicon,c,g,label)

        calibration = {k:result(raw,k!="ambiguous") for k,raw in cal.items()}
        fitted = calibrate(calibration["clear"],calibration["ambiguous"])
        rows = {k:result(raw,k!="ambiguous") for k,raw in test.items()}
        results.append(dict(mode=mode,calibration=calibration,gate=fitted,rows=rows,
                            lexicon={w:lexicon[vocabulary[w]].tolist() for w in verbs}))
    return results
