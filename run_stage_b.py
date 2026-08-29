import copy
import json
import random
import statistics
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import app


OUT = Path("stage-b-output")
OUT.mkdir(exist_ok=True)
N_MEMORIES = 6
N_TRAIN = 800
N_TEST = 240
D_MODEL = 64
EPOCHS = 12
BATCH = 64

random.seed(app.SEED)
torch.manual_seed(app.SEED)
torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


def make_cases(n, offset):
    return [
        app.make_case(random.Random(app.SEED + 100_000_003 + 104729 * (offset + i)), N_MEMORIES)
        for i in range(n)
    ]


def teacher_scores(cases):
    pairs = []
    mapping = []
    for ci, case in enumerate(cases):
        for mi, memory in enumerate(case.memories):
            pairs.append((f"Context: {memory}\nQuestion: ", case.query))
            mapping.append((ci, mi))
    t0 = time.perf_counter()
    nlls = app.continuation_nll_batch(pairs)
    elapsed = time.perf_counter() - t0
    rows = [[0.0] * N_MEMORIES for _ in cases]
    for nll, (ci, mi) in zip(nlls, mapping):
        rows[ci][mi] = -nll
    return torch.tensor(rows, dtype=torch.float32), elapsed


def make_tensor(texts):
    encoded = [app._encode_segment(x) for x in texts]
    width = max(len(x) for x in encoded)
    pad = app.tokenizer.pad_token_id
    out = torch.full((len(encoded), width), pad, dtype=torch.long)
    mask = torch.zeros((len(encoded), width), dtype=torch.bool)
    for i, seq in enumerate(encoded):
        out[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        mask[i, :len(seq)] = True
    return out, mask


def tensorize(cases):
    q, qmask = make_tensor([c.query for c in cases])
    flat_mem = [m for c in cases for m in c.memories]
    m, mmask = make_tensor(flat_mem)
    m = m.reshape(len(cases), N_MEMORIES, -1)
    mmask = mmask.reshape(len(cases), N_MEMORIES, -1)
    y = torch.tensor([c.label for c in cases], dtype=torch.long)
    return q, qmask, m, mmask, y


class QKRouter(nn.Module):
    def __init__(self, vocab_size, pad_id, d_model=D_MODEL):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.scale = d_model ** -0.5

    def pool(self, ids, mask):
        e = self.emb(ids)
        w = mask.to(e.dtype).unsqueeze(-1)
        return (e * w).sum(dim=-2) / w.sum(dim=-2).clamp(min=1)

    def forward(self, q, qmask, m, mmask):
        qh = self.q_proj(self.pool(q, qmask))
        mh = self.k_proj(self.pool(m, mmask))
        return torch.einsum("bd,bmd->bm", qh, mh) * self.scale


def train(model, mode, q, qmask, m, mmask, y, teacher):
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    order = list(range(len(y)))
    history = []
    model.train()
    for epoch in range(EPOCHS):
        random.Random(app.SEED + epoch).shuffle(order)
        losses = []
        for start in range(0, len(order), BATCH):
            ix = torch.tensor(order[start:start + BATCH], dtype=torch.long)
            logits = model(q[ix], qmask[ix], m[ix], mmask[ix])
            if mode == "direct":
                loss = F.cross_entropy(logits, y[ix])
            else:
                t = teacher[ix]
                # Query difficulty changes absolute NLL scale. Normalize within
                # each candidate set so the student learns relative compression utility.
                t = (t - t.mean(dim=-1, keepdim=True)) / (t.std(dim=-1, keepdim=True) + 1e-6)
                target = F.softmax(t, dim=-1)
                loss = -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
        history.append(statistics.mean(losses))
        print(f"{mode} epoch {epoch + 1}/{EPOCHS}: {history[-1]:.4f}", flush=True)
    return history


@torch.inference_mode()
def student_scores(model, tensors):
    q, qmask, m, mmask, _ = tensors
    model.eval()
    t0 = time.perf_counter()
    rows = []
    for start in range(0, len(q), 128):
        rows.append(model(q[start:start+128], qmask[start:start+128], m[start:start+128], mmask[start:start+128]))
    return torch.cat(rows, dim=0), time.perf_counter() - t0


def rank_metrics(scores, labels):
    order = torch.argsort(scores, dim=-1, descending=True)
    ranks = []
    picks = order[:, 0]
    for i, label in enumerate(labels.tolist()):
        ranks.append(int((order[i] == label).nonzero(as_tuple=False)[0].item()) + 1)
    return {
        "top1": statistics.mean(r == 1 for r in ranks),
        "top3": statistics.mean(r <= 3 for r in ranks),
        "mrr": statistics.mean(1 / r for r in ranks),
        "median_rank": statistics.median(ranks),
        "picks": picks.tolist(),
        "ranks": ranks,
    }


def answer_nll_for_picks(cases, picks):
    items = [
        (f"Context: {c.memories[p]}\nQuestion: {c.query}\nAnswer: ", c.answer)
        for c, p in zip(cases, picks)
    ]
    return statistics.mean(app.continuation_nll_batch(items))


print(f"Generating {N_TRAIN} train and {N_TEST} held-out test cases", flush=True)
train_cases = make_cases(N_TRAIN, 0)
test_cases = make_cases(N_TEST, 10000)

print("Computing SmolLM2 CDL teacher scores", flush=True)
teacher_train, teacher_train_seconds = teacher_scores(train_cases)
teacher_test, teacher_test_seconds = teacher_scores(test_cases)

train_tensors = tensorize(train_cases)
test_tensors = tensorize(test_cases)
qtr, qmasktr, mtr, mmasktr, ytr = train_tensors
qte, qmaskte, mte, mmaskte, yte = test_tensors

base = QKRouter(len(app.tokenizer), app.tokenizer.pad_token_id)
initial = copy.deepcopy(base.state_dict())
direct = QKRouter(len(app.tokenizer), app.tokenizer.pad_token_id)
direct.load_state_dict(initial)
distilled = QKRouter(len(app.tokenizer), app.tokenizer.pad_token_id)
distilled.load_state_dict(initial)

print("Training direct-label Q/K baseline", flush=True)
direct_history = train(direct, "direct", qtr, qmasktr, mtr, mmasktr, ytr, teacher_train)
print("Training CDL-distilled Q/K router", flush=True)
distill_history = train(distilled, "distilled", qtr, qmasktr, mtr, mmasktr, ytr, teacher_train)

direct_test, direct_seconds = student_scores(direct, test_tensors)
distill_test, distill_seconds = student_scores(distilled, test_tensors)
gzip_test = torch.tensor([app.gzip_scores(c) for c in test_cases], dtype=torch.float32)

methods = [
    ("gzip conditional", gzip_test),
    ("SmolLM2 CDL teacher", teacher_test),
    ("direct-label Q/K student", direct_test),
    ("CDL-distilled Q/K student", distill_test),
]

summary = []
details = []
teacher_choice = torch.argmax(teacher_test, dim=-1)
for name, scores in methods:
    met = rank_metrics(scores, yte)
    nll = answer_nll_for_picks(test_cases, met["picks"])
    agreement = statistics.mean(
        int(p == t) for p, t in zip(met["picks"], teacher_choice.tolist())
    )
    summary.append({
        "method": name,
        "top1": met["top1"],
        "top3": met["top3"],
        "MRR": met["mrr"],
        "median_rank": met["median_rank"],
        "teacher_choice_agreement": agreement,
        "selected_answer_NLL": nll,
    })
    for i, (pick, rank) in enumerate(zip(met["picks"], met["ranks"])):
        details.append({
            "case": i,
            "method": name,
            "label": test_cases[i].label,
            "pick": pick,
            "rank": rank,
            "correct": pick == test_cases[i].label,
            "query": test_cases[i].query,
            "answer": test_cases[i].answer,
            "relevant_memory": test_cases[i].memories[test_cases[i].label],
            "picked_memory": test_cases[i].memories[pick],
        })

oracle_nll = statistics.mean(app.continuation_nll_batch([
    (f"Context: {c.memories[c.label]}\nQuestion: {c.query}\nAnswer: ", c.answer)
    for c in test_cases
]))
query_only_nll = statistics.mean(app.continuation_nll_batch([
    (f"Question: {c.query}\nAnswer: ", c.answer)
    for c in test_cases
]))
summary.extend([
    {
        "method": "oracle relevant memory",
        "top1": 1.0, "top3": 1.0, "MRR": 1.0, "median_rank": 1.0,
        "teacher_choice_agreement": float("nan"), "selected_answer_NLL": oracle_nll,
    },
    {
        "method": "query only",
        "top1": float("nan"), "top3": float("nan"), "MRR": float("nan"), "median_rank": float("nan"),
        "teacher_choice_agreement": float("nan"), "selected_answer_NLL": query_only_nll,
    },
])

summary_df = pd.DataFrame(summary)
details_df = pd.DataFrame(details)
summary_df.to_csv(OUT / "summary.csv", index=False)
details_df.to_csv(OUT / "case_details.csv", index=False)
torch.save({"state_dict": direct.state_dict(), "config": {"d_model": D_MODEL}}, OUT / "direct_qk.pt")
torch.save({"state_dict": distilled.state_dict(), "config": {"d_model": D_MODEL}}, OUT / "distilled_qk.pt")
(OUT / "training_history.json").write_text(json.dumps({
    "direct": direct_history,
    "distilled": distill_history,
}, indent=2))
(OUT / "metadata.json").write_text(json.dumps({
    "model": app.MODEL_ID,
    "seed": app.SEED,
    "train_cases": N_TRAIN,
    "test_cases": N_TEST,
    "memories": N_MEMORIES,
    "student_d_model": D_MODEL,
    "epochs": EPOCHS,
    "teacher_train_seconds": teacher_train_seconds,
    "teacher_test_seconds": teacher_test_seconds,
    "direct_student_test_seconds": direct_seconds,
    "distilled_student_test_seconds": distill_seconds,
    "device": app.DEVICE,
}, indent=2))

print("\n=== Stage B: CDL distillation ===")
print(summary_df.to_string(index=False))
