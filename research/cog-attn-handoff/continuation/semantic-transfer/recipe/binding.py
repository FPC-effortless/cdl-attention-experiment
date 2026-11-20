"""Separable roles and operators on constructed inventory commands."""
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
from torch import nn

from language import NAMES, Grounder, apply, encode, fit

torch.set_num_threads(1)
KNOWN = ("copy", "add", "subtract", "maximize")
EXTRA = ("replicate", "augment", "debit", "floor")
PHRASES = (
    ("take the amount from {s}", "read the count in {s}", "{s} supplies the amount"),
    ("put the result into {d}", "update the balance in {d}", "{d} receives the result"),
    ("leave {u} alone", "ignore the balance in {u}", "{u} stays unchanged"),
)
COHORTS = ("familiar", "lexical", "layout", "joint", "implicit", "ambiguous")


def words_for_seed(seed):
    extra = list(EXTRA)
    random.Random(seed + 153).shuffle(extra)
    return tuple(extra)


def rows(seed, extra, n, cohort):
    rng = random.Random(seed)
    data = []
    for index in range(n):
        op = index % 4 if cohort == "support" else rng.randrange(4)
        s, d, u = rng.sample(NAMES, 3)
        phr = [rng.choice(p).format(s=s,d=d,u=u) for p in PHRASES]
        verb = (extra if cohort in ("support","lexical","joint","implicit","ambiguous") else KNOWN)[op]
        if cohort in ("familiar","support","lexical"):
            # Seen layouts present three different source/destination orderings.
            order = rng.choice(((0,1,2), (1,2,0), (2,0,1)))
            a,b,c = [phr[i] for i in order]
            text = f"{verb} stock . {a} ; {b} ; {c} ."
        elif cohort in ("layout","joint"):
            order = rng.choice(((2,1,0), (0,2,1), (1,0,2)))
            a,b,c = [phr[i] for i in order]
            text = rng.choice((f"For this reconciliation , {a} . then {b} . {c} . apply {verb} .",
                               f"Warehouse note : {a} , and after the count {b} , while {c} . method {verb} ."))
        elif cohort == "implicit":
            text = rng.choice((
                f"{u} is irrelevant to this ticket . whatever {s} holds is the quantity to {verb} into {d} .",
                f"The revised figure belongs on {d} using {verb} . the reference is whatever is sitting on {s} , not {u} .",
                f"{u} can wait . use the {verb} method for {d} with the number currently recorded under {s} ."))
        else:
            text = rng.choice((
                f"Use either {s} or {u} for the amount to {verb} into {d} .",
                f"{verb} from {s} , but I cannot tell whether {d} or {u} should receive it ."))
        tokens = re.findall(r"[a-z]+|[^\w\s]", text.lower())
        entities = list(dict.fromkeys(t for t in tokens if t in NAMES))
        src,dst = entities.index(s),entities.index(d)
        state = np.array([rng.randrange(8) for _ in range(3)],dtype=np.int64)
        target = state.copy()
        target[dst] = apply(op,state[src],state[dst])
        data.append((tokens,entities,state,target,op,src,dst))
    return data


def lexical_ids(raw, vocabulary, verb_words):
    result = []
    for tokens,*_ in raw:
        matches = [vocabulary[t] for t in tokens if t in verb_words]
        if len(matches) != 1:
            raise ValueError("expected one verb token in the declared vocabulary")
        result.append(matches[0])
    return torch.tensor(result)


class Roles(nn.Module):
    def __init__(self, vocabulary, entities, verbs, local):
        super().__init__()
        self.emb = nn.Embedding(vocabulary,32,padding_idx=0)
        self.entity_ids = entities
        self.register_buffer("verbs",torch.tensor(verbs))
        self.local = local
        if local:
            self.scorer = nn.Sequential(nn.Linear(11*32,96),nn.GELU(),nn.Linear(96,2))
        else:
            self.encoder = nn.GRU(32,32,batch_first=True,bidirectional=True)
            self.scorer = nn.Linear(64,2)

    def forward(self,x,actions):
        # Every entity has the same embedding in the masked role sequence.
        # The original mention IDs are used only to gather scores.
        masked = x.masked_fill(torch.isin(x,self.verbs),2)
        for entity in self.entity_ids:
            masked = masked.masked_fill(masked==entity,3)
        e = self.emb(masked)
        if self.local:
            windows = nn.functional.pad(e,(0,0,5,5)).unfold(1,11,1).flatten(2)
            token = self.scorer(windows)
        else:
            token = self.scorer(self.encoder(e)[0])
        mention = torch.stack([((x==t).unsqueeze(-1)*token).sum(1)/(x==t).sum(1,keepdim=True)
                               for t in self.entity_ids],dim=1)
        _,src,dst = actions.T
        return mention[:,src,0]+mention[:,dst,1]


def train_roles(model,data,steps,rng):
    x,compatible,_,actions = data
    optimizer = torch.optim.AdamW(model.parameters(),lr=.003)
    # No operation ID in this objective. Sum over every compatible operator per binding.
    role_compatible = compatible.reshape(-1,4,6).float().mean(1)
    for _ in range(steps):
        ix = torch.randint(len(x),(48,),generator=rng)
        logits = model(x[ix],actions[:6]).log_softmax(-1)
        loss = -(logits.exp()*role_compatible[ix]).sum(1).clamp_min(1e-12).log().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def learn_words(model,data,verb_ids,words):
    x,compatible,_,actions = data
    candidate = compatible.reshape(-1,4,6)
    with torch.inference_mode():
        role_dist = model(x,actions[:6]).softmax(-1)
        evidence = (role_dist[:,None,:]*candidate).sum(-1)
    # A small contamination floor keeps one unmodeled example from zeroing a whole word.
    likelihood = (.98*evidence+.02/512).log()
    return {token:likelihood[verb_ids==token].sum(0).softmax(-1) for token in words}


def batches(model,data,verb_ids=None,lexicon=None):
    x,_,_,actions = data
    model.eval()
    with torch.inference_mode():
        if lexicon is None:
            return torch.cat([model(v,actions).softmax(-1) for v in x.split(64)])
        role = torch.cat([model(v,actions[:6]).softmax(-1) for v in x.split(64)])
        op = torch.stack([lexicon[int(i)] for i in verb_ids])
        return (op[:,:,None]*role[:,None,:]).flatten(1)


def evaluate(model,data,cohort,verb_ids=None,lexicon=None):
    _,compatible,gold,actions = data
    p = batches(model,data,verb_ids,lexicon)
    confidence,pred = p.max(-1)
    correct = pred==gold
    outcome = compatible[torch.arange(len(p)),pred]
    action_pred,action_gold = actions[pred],actions[gold]
    return dict(cohort=cohort,exact=None if cohort=="ambiguous" else int(correct.sum()),
                effect=None if cohort=="ambiguous" else int(outcome.sum()),
                role=None if cohort=="ambiguous" else int((action_pred[:,1:]==action_gold[:,1:]).all(1).sum()),
                operation=None if cohort=="ambiguous" else int((action_pred[:,0]==action_gold[:,0]).sum()),
                accepted=int((confidence>=.8).sum()),
                correct_accepted=None if cohort=="ambiguous" else int((correct&(confidence>=.8)).sum()),
                predictions=pred.tolist(),confidence=confidence.tolist())


def run(seed,arm):
    torch.manual_seed(seed)
    extra = words_for_seed(seed)
    train = rows(seed,extra,4096,"familiar")
    support = rows(seed+3060,extra,96,"support")
    vocab = {v:i+4 for i,v in enumerate(sorted({w for r in train+support for w in r[0] if w not in NAMES}))}
    entities = list(range(len(vocab)+4,len(vocab)+7))
    word_ids = [vocab[w] for w in KNOWN+extra]
    tests = {c:rows(seed+7100+i,extra,512,c) for i,c in enumerate(COHORTS)}
    result = []
    modes = ("local","sequence") if arm=="candidate" else ("pointer","transformer")
    for mode in modes:
        torch.manual_seed(seed)
        rng = torch.Generator().manual_seed(seed+210)
        familiar = encode(train,vocab,entities)
        adapted = encode(train[:288]+support,vocab,entities)
        if mode in ("local","sequence"):
            model = Roles(len(vocab)+7,entities,word_ids,mode=="local")
            train_roles(model,familiar,650,rng)
            calibration = encode(train[:384]+support,vocab,entities)
            lexicon = learn_words(model,calibration,lexical_ids(train[:384]+support,vocab,KNOWN+extra),word_ids)
        else:
            model = Grounder(mode,len(vocab)+7,entities)
            fit(model,familiar,650,rng,.003)
            fit(model,adapted,180,rng,.0015)
            lexicon = None
        dest = Path(os.environ["TRAINLAB_OUTPUT_DIR"])/f"{mode}.npz"
        np.savez_compressed(dest,**{k:v.detach().numpy() for k,v in model.state_dict().items()})
        result.append(dict(mode=mode,rows=[evaluate(model,encode(raw,vocab,entities),c,
                           lexical_ids(raw,vocab,KNOWN+extra),lexicon) for c,raw in tests.items()],
                           lexicon=None if lexicon is None else {w:lexicon[vocab[w]].tolist() for w in KNOWN+extra}))
    return result
