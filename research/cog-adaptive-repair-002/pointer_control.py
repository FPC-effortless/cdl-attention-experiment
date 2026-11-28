"""Trainable extractive evidence-position control on BALANCE raw byte prompts.

The selector examines the full observed prompt when a query is complete. Each
candidate byte has a two-sided neighborhood; the query representation uses the
last eight observed bytes. Both the scoring network and relative locality weights
are learned. Output is the byte at the most likely location, so this control
cannot synthesize a transformed value. It is a task-specific, full-prefix reader.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from panels import DEV_PANELS, _batch_bytes, _english_world, _rng, _world, DIGITS


class Pointer(nn.Module):
    def __init__(self, wide=False, trigrams=None):
        super().__init__()
        dim = 12
        self.wide = wide
        self.trigrams = wide if trigrams is None else trigrams
        self.emb = nn.Embedding(256, dim)
        self.query_weight = nn.Parameter(torch.zeros(8))
        self.local = nn.Conv1d(dim + (2 if self.trigrams else 1), 24, 49 if wide else 21, padding=24 if wide else 10)
        self.query = nn.Linear(8 * dim, 24)
        self.offsets = nn.Parameter(torch.zeros(33))
        self.out = nn.Sequential(nn.Linear(50, 32), nn.GELU(), nn.Linear(32, 1))

    def score(self, ids, lengths):
        pos = torch.arange(ids.size(1), device=ids.device)
        valid = pos[None] < lengths[:, None]
        end = lengths[:, None] - 1 - torch.arange(7, -1, -1, device=ids.device)
        suffix = ids.gather(1, end.clamp_min(0))
        emb = self.emb(ids) * valid.unsqueeze(-1)
        q = self.query(self.emb(suffix).flatten(1)).tanh()
        # Learned query-byte salience drives a local match feature independent
        # of the identities seen in training. Every observed byte is eligible.
        match = ((ids[..., None] == suffix[:, None]).float() * self.query_weight.softmax(0)).sum(-1)
        match = match * valid
        features = [emb, match[..., None]]
        if self.trigrams:
            # Shared exact three-byte comparisons generalize across byte identities.
            # A learned distribution chooses useful trigrams in the query suffix.
            grams = F.pad(ids, (2, 0), value=-1).unfold(1, 3, 1)
            qgrams = suffix.unfold(1, 3, 1)
            equal = grams[:, :, None].eq(qgrams[:, None]).all(-1).float()
            salient = (equal * self.query_weight[2:].softmax(0)).sum(-1)
            features.append((salient * valid)[..., None])
        ctx = self.local(torch.cat(features, -1).transpose(1, 2)).transpose(1, 2).tanh()
        neighbors = F.pad(match, (16, 16)).unfold(-1, 33, 1)
        overlap = torch.logsumexp(6 * neighbors + self.offsets.log_softmax(0), -1) / 6
        feat = torch.cat((ctx, ctx * q[:, None], overlap[..., None], match[..., None]), -1)
        return self.out(feat).squeeze(-1).masked_fill(~valid, -torch.inf)

    def select(self, prompts):
        ids = _batch_bytes(prompts)
        lengths = torch.tensor(list(map(len, prompts)))
        scores = self.score(ids, lengths)
        indices = scores.argmax(1)
        return ids.gather(1, indices[:, None]).flatten().tolist(), indices.tolist()


def draw(rng, panel, template=0):
    if panel.startswith("english"):
        return _english_world(rng, panel, template)
    return _world(rng, panel)


def train(model, seed, objective, updates, batch, balanced=False):
    opt = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    for step in range(updates):
        rng = _rng(seed, step + 2**34, "train")
        panel = ("retrieval", "conflict", "english_revision","english_revision")[step % 4]
        template = int(rng.integers(2))
        if balanced and panel.startswith("english"):
            from repair_objective import balanced_english_world
            episodes = [balanced_english_world(rng, template, "fact") for _ in range(batch)]
        else:
            episodes = [draw(rng, panel, template) for _ in range(batch)]
        samples, indices, labels = [], [], []
        for before, after, old, new in episodes:
            diffs = [i for i,(a,b) in enumerate(zip(before,after)) if a!=b]
            assert len(diffs)==1 and before[diffs[0]]==old and after[diffs[0]]==new
            for prompt, value in [(before, old),(after,new)]:
                samples.append(prompt)
                indices.append(diffs[0])
                labels.append(value)
        ids = _batch_bytes(samples)
        scores = model.score(ids, torch.tensor(list(map(len, samples))))
        if objective == "position":
            loss = F.cross_entropy(scores, torch.tensor(indices))
        else:
            same = ids == torch.tensor(labels)[:,None]
            loss = (scores.logsumexp(1) - scores.masked_fill(~same,-torch.inf).logsumexp(1)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


def evaluate(model, worlds=128, seed=610003):
    out = {}
    for j,panel in enumerate(DEV_PANELS):
        rng = _rng(seed,j,"dev")
        episodes = [draw(rng,panel,2 if panel=="english_held_paraphrase" else i%2) for i in range(worlds)]
        count = {"worlds":worlds,"correct":0,"both_correct":0,"answers_changed":0,
                 "both_decisive_positions":0,"decisive_pairs":0}
        for start in range(0,worlds,16):
            block = episodes[start:start+16]
            with torch.no_grad():
                pred, pos = model.select([x for a,b,_,_ in block for x in (a,b)])
            for i,(a,b,y,z) in enumerate(block):
                p1,p2=pred[2*i:2*i+2]
                count["correct"]+=int(p1==y)+int(p2==z)
                count["both_correct"]+=int(p1==y and p2==z)
                count["answers_changed"]+=int(p1!=p2)
                if y!=z:
                    loc = next(k for k,(o,n) in enumerate(zip(a,b)) if o!=n)
                    count["both_decisive_positions"]+=int(pos[2*i]==loc==pos[2*i+1])
                    count["decisive_pairs"]+=1
        out[panel]=count
    return out



# Byte offsets are into the checked TRACE development corpus. The paired
# questions are manually written; passage words are naturally occurring.
# Each anchor identifies one source byte for evaluation only.
NATURAL = [
    (11261,140,("1 /",0,"What is the numerator of the stated pint fraction?"),("2 pint",0,"What is its denominator?")),
    (232153,195,("2 :",0,"What is the first term of the printed piece ratio?"),("1 ratio",0,"What is the second term of the printed piece ratio?")),
    (369741,161,("2 )",0,"Which DVD region does the text assign to Japan?"),("3 )",0,"Which DVD region does the text assign to Malaysia?")),
    (383338,143,("2 :",0,"What number is given first in the orbital resonance?"),("1 orbital",0,"What number is given second in the orbital resonance?")),
    (492551,83,("at 0",3,"Which number begins this sequence?"),("3 ,",0,"What number follows the starting number?")),
    (605329,106,("1 January",0,"On which day of January was Gauci born?"),("2 @-@",0,"How many classification points is the player described as having?")),
    (1164485,193,("1 ,",0,"On which September day did the storm become a weak tropical depression?"),("3 as",0,"On which September day did it become a typhoon?")),
    (1611808,139,("1 ",0,"What is the first number in the estimate in weeks?"),("2 weeks",0,"What is the second number in the estimate in weeks?")),
    (2272332,60,("2 to",0,"How many points did Ellis score?"),("0 at",0,"How many points did Betterman score?")),
    (2485725,137,("Yankees 2",8,"How many runs did the winning team score?"),("1 .",0,"How many runs did the Yankees score?")),
    (1720228,86,("NOK 1",4,"What is the first digit in the stated construction cost?"),("Package 2",8,"Which Oslo package financed the line?")),
]


# Keep scores, dates, denominators and construction-cost prefixes plausible
# after each synthetic edit. All answers remain inside the training digit set.
NATURAL_EDITS = {
    11261: b"33", 232153: b"33", 369741: b"11", 383338: b"33",
    492551: b"12", 605329: b"33", 1164485: b"22", 1611808: b"03",
    2272332: b"31", 2485725: b"30", 1720228: b"33",
}


def natural(model, root):
    data = np.memmap(root / "dev.bin",dtype=np.uint8,mode="r")
    rows=[]
    for offset,length,*tasks in NATURAL:
        text=bytes(data[offset:offset+length])
        pos=[text.index(anchor.encode("ascii"))+start for anchor,start,_ in tasks]
        values=[text[i] for i in pos]
        assert values[0] != values[1] and all(v in DIGITS for v in values)
        variants=[text]
        for target, replacement in zip(pos, NATURAL_EDITS[offset]):
            b=bytearray(text)
            assert replacement in DIGITS and replacement != b[target]
            b[target]=replacement
            variants.append(bytes(b))
        cases=[v+b" "+t[2].encode("ascii") for v in variants for t in tasks]
        gold=[v[i] for v in variants for i in pos]
        with torch.no_grad():
            pred, positions=model.select(cases)
        rows.append(dict(offset=offset,all_six_correct=pred==gold,
                         original_query_pair=pred[:2]==gold[:2],
                         selected_support=sum(p==pos[i%2] for i,p in enumerate(positions)),
                         answers_correct=sum(a==b for a,b in zip(gold,pred)),questions=6))
    return rows


def confirmation(model):
    from confirm_repair import PANELS, SEED, _counts, _episodes
    metrics={}
    with torch.no_grad():
        for panel in PANELS:
            episodes=_episodes(panel,128,SEED)
            decisions=[]
            for i in range(0,len(episodes),8):
                prompts=[x for before,after,_,_ in episodes[i:i+8] for x in (before,after)]
                pred,_=model.select(prompts)
                decisions+=pred
            metrics[panel]=_counts(episodes,decisions)
    return metrics


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--seed",type=int,required=True)
    ap.add_argument("--updates",type=int,default=384)
    ap.add_argument("--batch",type=int,default=12)
    ap.add_argument("--objective",choices=("position","answer"),required=True)
    ap.add_argument("--wide",action="store_true")
    ap.add_argument("--trigrams",action=argparse.BooleanOptionalAction,default=None)
    ap.add_argument("--balanced",action="store_true")
    ap.add_argument("--corpus",type=Path,default=Path("trace_corpus"))
    args=ap.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model=Pointer(wide=args.wide,trigrams=args.trigrams)
    before=evaluate(model,worlds=128)
    train(model,args.seed,args.objective,args.updates,args.batch,args.balanced)
    model.eval()
    after=evaluate(model)
    args.out.mkdir(exist_ok=True,parents=True)
    prefix=("wide-" if args.wide else "")
    if args.trigrams is not None and args.trigrams != args.wide:
        prefix += "add-grams-" if args.trigrams else "no-grams-"
    filename=prefix+("balanced-" if args.balanced else "")+f"{args.objective}-{args.seed}"
    torch.save(model.state_dict(),args.out/(filename+".pt"))
    (args.out/(filename+".json")).write_text(json.dumps(dict(
        seed=args.seed,updates=args.updates,worlds_per_update=args.batch,
        objective=args.objective,wide=args.wide,trigrams=model.trigrams,natural_version="3.0",parameters=sum(p.numel() for p in model.parameters()),
        initial=before,final=after,balanced=args.balanced,confirmation=confirmation(model),
        natural=natural(model,args.corpus)),indent=2)+"\n")
