"""LRR-009: exact-example-multiset curriculum intervention, no new losses.

All schedules consume every bank minibatch once clean and once augmented.
Only order changes. The original hard routing implementation is inherited.
"""
from __future__ import annotations
import hashlib, json, time
from collections import Counter
from pathlib import Path
import torch
from .lrr005_register import ID, vectorize
from .lrr008_invariance import (
    InvariantRegister, augmented_batch, objective, state_digest,
    heldout, hard_batch,
)
ARMS = ("interleaved", "clean_first", "augmentation_first")

def schedule(arm: str, bank_size: int = 500):
    if bank_size < 1 or arm not in ARMS:
        raise ValueError("invalid arm or bank size")
    clean = [(i,"clean") for i in range(bank_size)]
    aug = [(i,"augmentation") for i in range(bank_size)]
    if arm == "interleaved":
        return [item for pair in zip(clean,aug) for item in pair]
    return clean+aug if arm == "clean_first" else aug+clean

def payload_hash(batch):
    h=hashlib.sha256()
    h.update(json.dumps([(e.words,e.answer) for e in batch[0]],separators=(",",":")).encode())
    for t in batch[1:]:
        h.update(str(tuple(t.shape)).encode()); h.update(str(t.dtype).encode())
        h.update(t.contiguous().numpy().tobytes())
    return h.hexdigest()

def make_bank(seed: int, bank_size=500, histories=8):
    bank=[augmented_batch(seed+2000,i,histories) for i in range(bank_size)]
    hashes=[payload_hash(b) for b in bank]
    return bank, hashes

def event_digest(events, hashes, *, sorted_events=False):
    ordered=sorted(events) if sorted_events else events
    h=hashlib.sha256()
    for idx,mode in ordered:
        h.update(f"{idx}:{mode}:{hashes[idx]}\n".encode())
    return h.hexdigest()

@torch.no_grad()
def predictions(model, examples, chunk=32):
    """Original batched hard execution plus hard relation-code diagnostics."""
    result=[]
    for start in range(0,len(examples),chunk):
        exs=examples[start:start+chunk]
        d=hard_batch(model,exs)
        x,_,_=vectorize(exs)
        h=model.token_features(x)
        gates=model.role(h).sigmoid().tolist(); codes=model.code(h).argmax(-1).tolist()
        for i,(ex,row) in enumerate(zip(exs,d)):
            relation=None
            for p in range(len(ex.words)-1):
                if gates[i][p][1]>.5: relation=codes[i][p]
            row={**row,"query_hard_relation_code":relation}
            result.append(row)
    return result

def summarize(diag):
    n=len(diag)
    return {"trials":n,"correct":sum(d['correct'] for d in diag),
            "recall":sum(d['correct'] for d in diag)/n,
            "failure_counts":dict(Counter(d['failure'] for d in diag)),
            "query_code_counts":dict(Counter(str(d['query_hard_relation_code']) for d in diag)),
            "occupied_query_codes":len({d['query_hard_relation_code'] for d in diag if d['query_hard_relation_code'] is not None}),
            "mean_collision_pairs":sum(d['collision_pairs'] for d in diag)/n,
            "mean_nonvalue_writes":sum(d['nonvalue_writes'] for d in diag)/n,
            "mean_writes":sum(d['writes'] for d in diag)/n,
            "gold_written_fraction":sum(d['gold_written'] for d in diag)/n}

def dev_evaluate(model, seed, step, trials=128):
    rows=[];case_records=[]
    for cond in ('regular','untrained_in_vocab'):
        exs=[heldout(130_000_000_000+seed*1_000_000+i,32,cond,split='dev') for i in range(trials)]
        diag=predictions(model,exs)
        rows.append({"step":step,"condition":cond,**summarize(diag)})
        for i,(ex,d) in enumerate(zip(exs,diag)):
            case_records.append({"step":step,"condition":cond,"case":i,"gold":ex.answer,**d})
    return rows,case_records

def fit_schedule(seed,arm,bank,hashes,outdir,dev_trials=128):
    torch.set_num_threads(1);torch.manual_seed(seed)
    model=InvariantRegister();optimizer=torch.optim.AdamW(model.parameters(),lr=.004)
    events=schedule(arm,len(bank));init=state_digest(model)
    trace=[];milestones=[];dev=[];dev_cases=[];actual_events=[]
    total=len(events);checkpoints=sorted(set((total//4,total//2,3*total//4,total)))
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();train_seconds=0.0
    rows,cases=dev_evaluate(model.eval(),seed,0,dev_trials);dev+=rows;dev_cases+=cases
    for update,(idx,mode) in enumerate(events,1):
        ex,x,y,q,mask,noise=bank[idx]
        model.train();optimizer.zero_grad(set_to_none=True)
        t=time.perf_counter()
        loss,metrics=objective(model,x,y,q,mask,noise,mode)
        if not torch.isfinite(loss):raise RuntimeError(f"nonfinite {arm} {seed} {update}")
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.0);optimizer.step()
        train_seconds+=time.perf_counter()-t;actual_events.append((idx,mode))
        if update%50==0 or update==total:
            trace.append({"step":update,"bank_index":idx,"input_mode":mode,"loss":loss.item(),**metrics})
        if update in checkpoints:
            model.eval()
            rows,cases=dev_evaluate(model,seed,update,dev_trials);dev+=rows;dev_cases+=cases
            counts=[int(s['step'].item()) for s in optimizer.state.values() if 'step' in s]
            assert set(counts)=={update}
            path=outdir/f"lrr009_{arm}_{seed}_step{update}.pt"
            record={"seed":seed,"arm":arm,"step":update,"model_tensor_sha256":state_digest(model),
                    "init_tensor_sha256":init,"optimizer_step_min":min(counts),"optimizer_step_max":max(counts)}
            torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"receipt":record},path)
            milestones.append({**record,"filename":path.name,"checkpoint_sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    assert Counter(actual_events)==Counter(schedule('interleaved',len(bank)))
    # No in-place input mutation allowed; all tensors audited after training.
    assert [payload_hash(b) for b in bank]==hashes
    receipt={"seed":seed,"arm":arm,"parameters":sum(p.numel() for p in model.parameters()),
             "updates":total,"clean_updates":sum(m=='clean' for _,m in events),
             "augmented_updates":sum(m=='augmentation' for _,m in events),
             "unique_histories":len(bank)*8,"question_view_exposures":total*48,
             "init_tensor_sha256":init,"bank_sha256":hashlib.sha256(''.join(hashes).encode()).hexdigest(),
             "event_multiset_sha256":event_digest(actual_events,hashes,sorted_events=True),
             "chronological_event_sha256":event_digest(actual_events,hashes),
             "update_wall_seconds":train_seconds,"fit_plus_dev_wall_seconds":time.perf_counter()-started,
             "trace":trace,"milestones":milestones,"dev":dev}
    assert receipt['parameters']==4748
    return model,receipt,dev_cases
