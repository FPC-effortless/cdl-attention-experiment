import gc
import gzip
import json
import math
import random
import statistics
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import gradio as gr
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import spaces
except ImportError:
    class _Spaces:
        @staticmethod
        def GPU(*args, **kwargs):
            def deco(fn):
                return fn
            return deco
    spaces = _Spaces()

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
SEED = 20260829

ENTITIES = [
    "Zoravia", "Nelmora", "Tavaryn", "Quessia", "Brinora", "Velmara", "Kordune", "Salenne",
    "Morveth", "Daxora", "Ilyra", "Pevorin", "Caldris", "Nurevia", "Talmera", "Vostin",
    "Eldara", "Ruvane", "Kesmir", "Jorath", "Lunessa", "Fendrel", "Arvona", "Mireth",
]
VALUES = [
    "Merin", "Toval", "Sera", "Kelm", "Orin", "Veyra", "Dorin", "Pellan",
    "Neris", "Cavor", "Luma", "Tarin", "Bressa", "Ovel", "Rennan", "Silva",
    "Morda", "Avelin", "Keris", "Torra", "Venn", "Iskar", "Belin", "Navor",
]

RELATIONS = {
    "capital": {
        "facts": [
            "The seat of government for {e} is {v}.",
            "{e} has its capital at {v}.",
            "The capital city of {e} is {v}.",
        ],
        "queries": [
            "What is the capital of {e}?",
            "Which city is the seat of government for {e}?",
            "Name the capital city of {e}.",
        ],
    },
    "currency": {
        "facts": [
            "The legal tender used in {e} is {v}.",
            "{e} uses {v} as its currency.",
            "The currency of {e} is {v}.",
        ],
        "queries": [
            "What currency does {e} use?",
            "What is the legal tender in {e}?",
            "Name the currency of {e}.",
        ],
    },
    "leader": {
        "facts": [
            "The head of government for {e} is {v}.",
            "{v} currently leads {e}.",
            "The leader of {e} is {v}.",
        ],
        "queries": [
            "Who leads {e}?",
            "Who is the head of government for {e}?",
            "Name the leader of {e}.",
        ],
    },
    "language": {
        "facts": [
            "The official tongue of {e} is {v}.",
            "People in {e} officially use {v}.",
            "The official language of {e} is {v}.",
        ],
        "queries": [
            "What is the official language of {e}?",
            "Which tongue is official in {e}?",
            "Name the official language used by {e}.",
        ],
    },
}

@dataclass
class Case:
    relation: str
    entity: str
    answer: str
    query: str
    memories: List[str]
    label: int


def _fact(rng: random.Random, relation: str, entity: str, value: str) -> str:
    return rng.choice(RELATIONS[relation]["facts"]).format(e=entity, v=value)


def _query(rng: random.Random, relation: str, entity: str) -> str:
    return rng.choice(RELATIONS[relation]["queries"]).format(e=entity)


def make_case(rng: random.Random, n_memories: int = 6) -> Case:
    relation = rng.choice(list(RELATIONS))
    entity = rng.choice(ENTITIES)
    answer = rng.choice(VALUES)
    relevant = _fact(rng, relation, entity, answer)
    query = _query(rng, relation, entity)
    memories = [relevant]

    r2 = rng.choice([r for r in RELATIONS if r != relation])
    memories.append(_fact(rng, r2, entity, rng.choice(VALUES)))

    e2 = rng.choice([e for e in ENTITIES if e != entity])
    memories.append(_fact(rng, relation, e2, rng.choice(VALUES)))

    r3 = rng.choice([r for r in RELATIONS if r != relation])
    e3 = rng.choice([e for e in ENTITIES if e != entity])
    memories.append(_fact(rng, r3, e3, answer))

    while len(memories) < n_memories:
        rr = rng.choice(list(RELATIONS))
        ee = rng.choice(ENTITIES)
        vv = rng.choice(VALUES)
        candidate = _fact(rng, rr, ee, vv)
        if candidate not in memories:
            memories.append(candidate)

    rng.shuffle(memories)
    return Case(relation, entity, answer, query, memories, memories.index(relevant))


def make_benchmark(n_cases: int, n_memories: int, seed: int = SEED) -> List[Case]:
    rng = random.Random(seed + 1009 * n_cases + 7919 * n_memories)
    return [make_case(rng, n_memories) for _ in range(n_cases)]


def gzip_len(text: str) -> int:
    return len(gzip.compress(text.encode("utf-8"), compresslevel=9))


def gzip_scores(case: Case) -> List[float]:
    q = case.query
    return [-(gzip_len(m + "\n" + q) - gzip_len(m)) for m in case.memories]


print(f"Loading {MODEL_ID} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    attn_implementation="eager",
)
model.eval().to(DEVICE)


def _encode_segment(text: str) -> List[int]:
    return tokenizer(text, add_special_tokens=False).input_ids


@torch.inference_mode()
def continuation_nll_batch(items: List[Tuple[str, str]], batch_size: int = 96) -> List[float]:
    outputs: List[float] = []
    pad_id = tokenizer.pad_token_id
    for start in range(0, len(items), batch_size):
        chunk = items[start:start + batch_size]
        seqs, cont_spans = [], []
        for prefix, continuation in chunk:
            p = _encode_segment(prefix)
            c = _encode_segment(continuation)
            if not p:
                p = [tokenizer.bos_token_id or tokenizer.eos_token_id]
            seq = p + c
            seqs.append(seq)
            cont_spans.append((len(p), len(c)))
        max_len = max(map(len, seqs))
        input_ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long, device=DEVICE)
        attention_mask = torch.zeros_like(input_ids)
        for i, seq in enumerate(seqs):
            input_ids[i, :len(seq)] = torch.tensor(seq, device=DEVICE)
            attention_mask[i, :len(seq)] = 1
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        logp = torch.log_softmax(logits.float(), dim=-1)
        for i, (p_len, c_len) in enumerate(cont_spans):
            vals = []
            seq = input_ids[i]
            for off in range(c_len):
                pos = p_len + off
                if pos == 0:
                    continue
                vals.append(-logp[i, pos - 1, seq[pos]].item())
            outputs.append(float(statistics.mean(vals)) if vals else float("inf"))
        del logits, logp, input_ids, attention_mask
    return outputs


@torch.inference_mode()
def native_attention_scores(cases: List[Case], max_cases: int = 80) -> Dict[int, List[float]]:
    result: Dict[int, List[float]] = {}
    for ci, case in enumerate(cases[:max_cases]):
        scores = []
        for memory in case.memories:
            prefix = f"Context: {memory}\nQuestion: "
            p = _encode_segment(prefix)
            q = _encode_segment(case.query)
            ids = torch.tensor([p + q], dtype=torch.long, device=DEVICE)
            mask = torch.ones_like(ids)
            out = model(input_ids=ids, attention_mask=mask, output_attentions=True, use_cache=False)
            att = out.attentions[-1].float()
            q_start = len(p)
            mass = att[0, :, q_start:, :q_start].sum(dim=-1).mean().item()
            scores.append(float(mass))
            del out, att, ids, mask
        result[ci] = scores
    return result


def rank_metrics(score_rows: List[List[float]], labels: List[int]) -> Dict[str, float]:
    ranks = []
    for scores, label in zip(score_rows, labels):
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        ranks.append(order.index(label) + 1)
    return {
        "top1": sum(r == 1 for r in ranks) / len(ranks),
        "top3": sum(r <= 3 for r in ranks) / len(ranks),
        "mrr": statistics.mean(1 / r for r in ranks),
        "median_rank": statistics.median(ranks),
    }


def selected_answer_nll(cases: List[Case], score_rows: List[List[float]]) -> float:
    items = []
    for case, scores in zip(cases, score_rows):
        chosen = max(range(len(scores)), key=lambda i: scores[i])
        prefix = f"Context: {case.memories[chosen]}\nQuestion: {case.query}\nAnswer: "
        items.append((prefix, case.answer))
    return statistics.mean(continuation_nll_batch(items))


def answer_nll_for_true(cases: List[Case]) -> float:
    items = [
        (f"Context: {c.memories[c.label]}\nQuestion: {c.query}\nAnswer: ", c.answer)
        for c in cases
    ]
    return statistics.mean(continuation_nll_batch(items))


def answer_nll_query_only(cases: List[Case]) -> float:
    return statistics.mean(continuation_nll_batch([(f"Question: {c.query}\nAnswer: ", c.answer) for c in cases]))


@spaces.GPU(duration=60)
def run_benchmark(n_cases: int, n_memories: int, include_native_attention: bool):
    n_cases = int(n_cases)
    n_memories = int(n_memories)
    cases = make_benchmark(n_cases, n_memories)
    labels = [c.label for c in cases]
    started = time.perf_counter()

    gz = [gzip_scores(c) for c in cases]

    pair_items, mapping = [], []
    for ci, c in enumerate(cases):
        for mi, m in enumerate(c.memories):
            pair_items.append((f"Context: {m}\nQuestion: ", c.query))
            mapping.append((ci, mi))
    nlls = continuation_nll_batch(pair_items)
    cdl = [[0.0] * n_memories for _ in cases]
    for nll, (ci, mi) in zip(nlls, mapping):
        cdl[ci][mi] = -nll

    rows = []
    for name, scores in [("Gzip conditional", gz), ("SmolLM2 conditional description length", cdl)]:
        m = rank_metrics(scores, labels)
        rows.append({
            "method": name,
            "top1": m["top1"], "top3": m["top3"], "MRR": m["mrr"],
            "median_rank": m["median_rank"],
            "selected_answer_NLL": selected_answer_nll(cases, scores),
        })

    if include_native_attention:
        cap = min(n_cases, 80)
        att_map = native_attention_scores(cases, max_cases=cap)
        att_scores = [att_map[i] for i in range(cap)]
        att_cases = cases[:cap]
        m = rank_metrics(att_scores, [c.label for c in att_cases])
        rows.append({
            "method": f"Native last-layer attention mass (n={cap})",
            "top1": m["top1"], "top3": m["top3"], "MRR": m["mrr"],
            "median_rank": m["median_rank"],
            "selected_answer_NLL": selected_answer_nll(att_cases, att_scores),
        })

    rows.append({
        "method": "Oracle relevant memory",
        "top1": 1.0, "top3": 1.0, "MRR": 1.0, "median_rank": 1.0,
        "selected_answer_NLL": answer_nll_for_true(cases),
    })
    rows.append({
        "method": "Query only (no retrieval)",
        "top1": float("nan"), "top3": float("nan"), "MRR": float("nan"), "median_rank": float("nan"),
        "selected_answer_NLL": answer_nll_query_only(cases),
    })

    result = pd.DataFrame(rows)
    elapsed = time.perf_counter() - started

    examples = []
    for i, c in enumerate(cases[:10]):
        gz_pick = max(range(n_memories), key=lambda j: gz[i][j])
        cdl_pick = max(range(n_memories), key=lambda j: cdl[i][j])
        examples.append({
            "relation": c.relation,
            "query": c.query,
            "answer": c.answer,
            "relevant_memory": c.memories[c.label],
            "gzip_pick": c.memories[gz_pick],
            "cdl_pick": c.memories[cdl_pick],
            "gzip_correct": gz_pick == c.label,
            "cdl_correct": cdl_pick == c.label,
        })

    metadata = {
        "model": MODEL_ID,
        "seed": SEED,
        "cases": n_cases,
        "memories_per_case": n_memories,
        "native_attention_requested": include_native_attention,
        "elapsed_seconds": elapsed,
        "device": DEVICE,
        "hypothesis": "Memory relevance can be estimated by -NLL(query | memory), i.e. conditional description length.",
    }
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result, pd.DataFrame(examples), json.dumps(metadata, indent=2)


with gr.Blocks(title="Conditional Description-Length Attention Test") as demo:
    gr.Markdown(
        "# Conditional Description-Length Attention Test\n"
        "Tests whether **SmolLM2 conditional description length** ranks relevant memories better than gzip and native attention, "
        "then checks whether the chosen memory actually lowers answer NLL."
    )
    with gr.Row():
        n_cases = gr.Slider(20, 250, value=120, step=20, label="Benchmark cases")
        n_memories = gr.Slider(4, 24, value=6, step=2, label="Candidate memories per case")
        include_attention = gr.Checkbox(value=True, label="Include native attention baseline (capped at 80 cases)")
    run = gr.Button("Run falsification benchmark", variant="primary")
    results = gr.Dataframe(label="Aggregate results")
    examples = gr.Dataframe(label="First 10 routing decisions")
    metadata = gr.Code(label="Run metadata", language="json")
    run.click(run_benchmark, [n_cases, n_memories, include_attention], [results, examples, metadata])

if __name__ == "__main__":
    demo.queue().launch()
