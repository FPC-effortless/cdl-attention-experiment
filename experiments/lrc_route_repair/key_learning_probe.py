"""Development-only learned key formation probe; no capability test claim.

Fit contrastive read/write keys from support indices, fit a shared codebook on
training writers, then distill readers into those codes. Evaluation reads a hard
index and scores only capped candidate contents. No semantic IDs enter forward.
This file is deliberately independent of the production model integration.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from lrc_vb001_comparison.source import study as vb


class LearnedKeys(nn.Module):
    def __init__(self, width=32, codes=16):
        super().__init__()
        self.encoder = vb.SentenceEncoder(width)
        self.writer = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
        self.first = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
        self.second = nn.Sequential(nn.Linear(2 * width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))
        self.register_buffer("codebook", torch.zeros(codes, width))

    def write(self, records):
        values = self.encoder(records)
        return values, F.normalize(self.writer(values), dim=-1)

    def query_first(self, query):
        state = self.encoder(query)
        return state, F.normalize(self.first(state), dim=-1)

    def query_second(self, state, value):
        return F.normalize(self.second(torch.cat((state, value), -1)), dim=-1)

    def training_keys(self, data):
        values, keys = self.write(data["records"])
        state, first = self.query_first(data["query"])
        first_value = values[torch.arange(len(values)), data["support"][:, 0]]
        second = self.query_second(state, first_value)
        return keys, torch.stack((first, second), 1)


def spherical_kmeans(points, count=16, seed=1, restarts=5, iterations=100):
    """Unsupervised codebook fitting; selects restarts by training inertia only."""
    generator = torch.Generator().manual_seed(seed)
    best = None
    for _ in range(restarts):
        chosen = [int(torch.randint(len(points), (), generator=generator))]
        for _ in range(count - 1):
            distance = (1 - points @ points[chosen].T).amin(-1).clamp_min(1e-8)
            distance[chosen] = 0
            chosen.append(int(torch.multinomial(distance, 1, generator=generator)))
        centers = points[chosen].clone()
        for _ in range(iterations):
            assignments = (points @ centers.T).argmax(-1)
            newer = torch.stack([
                points[assignments == i].mean(0) if (assignments == i).any() else centers[i]
                for i in range(count)
            ])
            newer = F.normalize(newer, dim=-1)
            if (newer - centers).abs().max() < 1e-6:
                centers = newer
                break
            centers = newer
        inertia = float((1 - (points @ centers.T).amax(-1)).mean())
        if best is None or inertia < best[0]:
            best = (inertia, centers)
    return best


@torch.no_grad()
def infer_bounded(model, records, query, probes=2, capacity=2):
    """No gold inputs. Hard address lookup, explicit overflow, capped scoring.

    The O(N) write/index build is performed once per supplied memory. A query's
    candidate scoring examines at most probes * capacity records per hop.
    The O(K) codebook lookup is independent of memory length N.
    """
    values, keys = model.write(records)
    codes = (keys @ model.codebook.T).argmax(-1)
    states, first = model.query_first(query)
    selected = torch.full((len(records), 2), -1, dtype=torch.long)
    counters = dict(scored=0, probes=0, overflow_probes=0, maximum_scored_per_hop=0)
    for row in range(len(records)):
        buckets = {}
        for position, code in enumerate(codes[row].tolist()):
            buckets.setdefault(code, []).append(position)
        read_key = first[row]
        for hop in range(2):
            requested = (read_key @ model.codebook.T).topk(probes).indices.tolist()
            candidates = []
            for code in requested:
                counters["probes"] += 1
                positions = buckets.get(code, [])
                if len(positions) > capacity:
                    counters["overflow_probes"] += 1
                else:
                    candidates.extend(positions)
            counters["scored"] += len(candidates)
            counters["maximum_scored_per_hop"] = max(counters["maximum_scored_per_hop"], len(candidates))
            if candidates:
                choice = int((keys[row, candidates] @ read_key).argmax())
                selected[row, hop] = candidates[choice]
            if hop == 0:
                value = values[row, selected[row, 0]] if selected[row, 0] >= 0 else torch.zeros_like(states[row])
                read_key = model.query_second(states[row], value)
    return selected, codes, counters


@torch.no_grad()
def evaluate(model, data, probes=2, capacity=2):
    model.eval()
    keys, read_keys = model.training_keys(data)
    # Oracle-first is an explicit representation diagnostic, never inference.
    dense = torch.einsum("bhd,bnd->bhn", read_keys, keys).argmax(-1)
    write_codes = (keys @ model.codebook.T).argmax(-1)
    read_codes = (read_keys @ model.codebook.T).argmax(-1)
    target_codes = write_codes.gather(1, data["support"])
    indices, codes, counters = infer_bounded(model, data["records"], data["query"], probes, capacity)
    hits = indices.eq(data["support"])
    return {
        "examples": len(indices),
        "oracle_first_dense_support_hit": dense.eq(data["support"]).float().mean(0).tolist(),
        "oracle_first_code_agreement": read_codes.eq(target_codes).float().mean(0).tolist(),
        "hard_index_support_hit": hits.float().mean(0).tolist(),
        "hard_index_both_hit": float(hits.all(-1).float().mean()),
        "occupied_codes_per_story": float(torch.tensor([len(set(row.tolist())) for row in codes]).float().mean()),
        "null_read_fraction": float(indices.lt(0).float().mean()),
        "counters": counters,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=84101)
    parser.add_argument("--steps", type=int, default=1600)
    parser.add_argument("--distill-steps", type=int, default=400)
    parser.add_argument("--positive-weight", type=float, default=1.0)
    parser.add_argument("--probes", type=int, default=2)
    parser.add_argument("--capacity", type=int, default=2)
    parser.add_argument("--bridge-source", type=Path, default=Path("/workspace/scratch/3f2444385d54/lrc_diagnosis_001/repair_pilot/bridge_data.py"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    train = vb.dataset(71001, 2048)
    development = {"dev_iid": vb.dataset(84001, 64), "dev_withheld": vb.dataset(84002, 64, True)}
    spec = importlib.util.spec_from_file_location("bridge_development", args.bridge_source)
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    development["dev_bridge"] = bridge.dataset(84003, 64)
    model = LearnedKeys()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0)
    generator = torch.Generator().manual_seed(args.seed + 10000)
    started = time.perf_counter()
    for step in range(args.steps):
        batch = vb.take(train, torch.randint(len(train["answer"]), (32,), generator=generator))
        keys, reads = model.training_keys(batch)
        similarity = torch.einsum("bhd,bnd->bhn", reads, keys)
        contrastive = F.cross_entropy((similarity / 0.08).flatten(0, 1), batch["support"].flatten())
        positive = 1 - similarity.gather(-1, batch["support"].unsqueeze(-1)).mean()
        loss = contrastive + args.positive_weight * positive
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        if (step + 1) % 400 == 0:
            print(json.dumps({"stage": "contrastive", "step": step + 1, "loss": float(loss.detach()), "contrastive": float(contrastive.detach()), "positive": float(positive.detach())}), flush=True)
    # Fit on observed training record strings only; no metadata or semantic IDs.
    strings = torch.unique(train["records"].flatten(0, 1), dim=0)
    with torch.no_grad():
        _, training_keys = model.write(strings)
        inertia, centers = spherical_kmeans(training_keys, seed=args.seed + 20000)
        model.codebook.copy_(centers)
    before = {name: evaluate(model, data, args.probes, args.capacity) for name, data in development.items()}
    print(json.dumps({"stage": "after_kmeans", "training_inertia": inertia, "results": before}), flush=True)
    # Freeze shared encoder and writer to prevent moving code labels. Fit only readers.
    for parameter in list(model.encoder.parameters()) + list(model.writer.parameters()):
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.002, weight_decay=0)
    for step in range(args.distill_steps):
        batch = vb.take(train, torch.randint(len(train["answer"]), (32,), generator=generator))
        keys, reads = model.training_keys(batch)
        target_codes = (keys @ model.codebook.T).argmax(-1).gather(1, batch["support"])
        code_logits = reads @ model.codebook.T / 0.08
        similarity = torch.einsum("bhd,bnd->bhn", reads, keys)
        loss = F.cross_entropy(code_logits.flatten(0, 1), target_codes.flatten()) + 0.25 * F.cross_entropy((similarity / 0.08).flatten(0, 1), batch["support"].flatten())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
    results = {name: evaluate(model, data, args.probes, args.capacity) for name, data in development.items()}
    print(json.dumps({"stage": "final_development_only", "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "parameters": sum(p.numel() for p in model.parameters()), "elapsed_seconds": time.perf_counter() - started, "data_hashes": {"train": vb.digest_data(train), **{k: vb.digest_data(v) for k, v in development.items()}}, "torch_version": str(torch.__version__), "training_unique_strings": len(strings), "training_inertia": inertia, "results": results}), flush=True)


if __name__ == "__main__":
    main()
