import json
import random
import statistics
from pathlib import Path

import pandas as pd
import torch

import app


OUT = Path("attention-control-output")
OUT.mkdir(exist_ok=True)
N_CASES = 120
N_MEMORIES = 6


def paired_make_benchmark(n_cases: int, n_memories: int, seed: int = app.SEED):
    return [
        app.make_case(random.Random(seed + 104729 * i), n_memories)
        for i in range(n_cases)
    ]


@torch.inference_mode()
def joint_attention_scores(cases):
    """Score all memories in one shared context, not one memory at a time.

    Returns three diagnostics:
    - last-layer attention mass to each memory span;
    - all-layer mean attention mass to each memory span;
    - maximum-head last-layer attention mass to each memory span.
    """
    last_rows = []
    all_rows = []
    max_head_rows = []

    for ci, case in enumerate(cases):
        ids = []
        spans = []
        for mi, memory in enumerate(case.memories):
            label_ids = app._encode_segment(f"Memory {mi + 1}: ")
            mem_ids = app._encode_segment(memory)
            newline_ids = app._encode_segment("\n")
            ids.extend(label_ids)
            start = len(ids)
            ids.extend(mem_ids)
            end = len(ids)
            spans.append((start, end))
            ids.extend(newline_ids)

        ids.extend(app._encode_segment("Question: "))
        q_start = len(ids)
        q_ids = app._encode_segment(case.query)
        ids.extend(q_ids)
        q_end = len(ids)

        input_ids = torch.tensor([ids], dtype=torch.long, device=app.DEVICE)
        mask = torch.ones_like(input_ids)
        out = app.model(
            input_ids=input_ids,
            attention_mask=mask,
            output_attentions=True,
            use_cache=False,
        )

        layers = [a[0].float() for a in out.attentions]
        last = layers[-1]  # [heads, seq, seq]

        last_scores = []
        all_scores = []
        max_head_scores = []
        for start, end in spans:
            # Natural attention mass: sum over memory tokens, then average query
            # positions and heads. All candidates coexist in the same softmax.
            per_head_last = last[:, q_start:q_end, start:end].sum(dim=-1).mean(dim=-1)
            last_scores.append(per_head_last.mean().item())
            max_head_scores.append(per_head_last.max().item())

            layer_masses = []
            for att in layers:
                mass = att[:, q_start:q_end, start:end].sum(dim=-1).mean()
                layer_masses.append(mass.item())
            all_scores.append(statistics.mean(layer_masses))

        last_rows.append(last_scores)
        all_rows.append(all_scores)
        max_head_rows.append(max_head_scores)

        if (ci + 1) % 20 == 0:
            print(f"Joint attention: {ci + 1}/{len(cases)} cases", flush=True)

        del out, input_ids, mask, layers, last

    return last_rows, all_rows, max_head_rows


def metrics(rows, labels):
    ranks = []
    picks = []
    for scores, label in zip(rows, labels):
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        picks.append(order[0])
        ranks.append(order.index(label) + 1)
    return {
        "top1": sum(r == 1 for r in ranks) / len(ranks),
        "top3": sum(r <= 3 for r in ranks) / len(ranks),
        "mrr": statistics.mean(1 / r for r in ranks),
        "median_rank": statistics.median(ranks),
        "picks": picks,
        "ranks": ranks,
    }


app.make_benchmark = paired_make_benchmark
cases = paired_make_benchmark(N_CASES, N_MEMORIES)
labels = [c.label for c in cases]

# Re-run the primary methods on exactly the same paired cases.
gzip_rows = [app.gzip_scores(c) for c in cases]
pair_items = []
mapping = []
for ci, c in enumerate(cases):
    for mi, memory in enumerate(c.memories):
        pair_items.append((f"Context: {memory}\nQuestion: ", c.query))
        mapping.append((ci, mi))
nlls = app.continuation_nll_batch(pair_items)
cdl_rows = [[0.0] * N_MEMORIES for _ in cases]
for nll, (ci, mi) in zip(nlls, mapping):
    cdl_rows[ci][mi] = -nll

last_rows, all_rows, max_head_rows = joint_attention_scores(cases)

method_rows = [
    ("gzip conditional", gzip_rows),
    ("SmolLM2 conditional description length", cdl_rows),
    ("joint attention: last-layer mean-head mass", last_rows),
    ("joint attention: all-layer mean mass", all_rows),
    ("joint attention: last-layer max-head mass", max_head_rows),
]

summary = []
details = []
for name, rows in method_rows:
    m = metrics(rows, labels)
    answer_nll = app.selected_answer_nll(cases, rows)
    summary.append({
        "method": name,
        "top1": m["top1"],
        "top3": m["top3"],
        "MRR": m["mrr"],
        "median_rank": m["median_rank"],
        "selected_answer_NLL": answer_nll,
    })
    for i, (pick, rank) in enumerate(zip(m["picks"], m["ranks"])):
        details.append({
            "case": i,
            "method": name,
            "label": labels[i],
            "pick": pick,
            "rank": rank,
            "correct": pick == labels[i],
            "query": cases[i].query,
            "answer": cases[i].answer,
            "relevant_memory": cases[i].memories[labels[i]],
            "picked_memory": cases[i].memories[pick],
        })

summary_df = pd.DataFrame(summary)
details_df = pd.DataFrame(details)
summary_df.to_csv(OUT / "summary.csv", index=False)
details_df.to_csv(OUT / "case_details.csv", index=False)
metadata = {
    "model": app.MODEL_ID,
    "seed": app.SEED,
    "cases": N_CASES,
    "memories": N_MEMORIES,
    "device": app.DEVICE,
    "design": "all candidate memories concatenated into a single context for attention scoring",
}
(OUT / "metadata.json").write_text(json.dumps(metadata, indent=2))
print("\n=== Joint-context attention control ===")
print(summary_df.to_string(index=False))
