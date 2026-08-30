from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Sequence

VERSION = "CASM-Bench-v1.0"
TASKS = ("assoc", "state", "arith", "rule", "graph", "reverse")
DEV_N_PER_TASK = 60
HOLDOUT_N_PER_TASK = 200
DEV_SEEDS = {"core": 0xC4510001, "ood": 0xC4510002}
HOLDOUT_SEEDS = {"core": 0xC4511001, "ood": 0xC4511002}


@dataclass(frozen=True)
class BenchCase:
    version: str
    suite: str
    task: str
    index: int
    prompt: str
    answer: str
    metadata: Mapping[str, object]

    @property
    def prompt_hash(self) -> str:
        return hashlib.sha256(self.prompt.encode("utf-8")).hexdigest()

    @property
    def case_id(self) -> str:
        blob = json.dumps({
            "version": self.version,
            "suite": self.suite,
            "task": self.task,
            "index": self.index,
            "prompt": self.prompt,
            "answer": self.answer,
        }, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()


def _names(n: int, prefix: str = "n") -> List[str]:
    return [f"{prefix}{i}" for i in range(n)]


def _assoc(rng: random.Random, *, ood: bool):
    n = 24 if ood else 12
    keys = _names(n, "k")
    mapping = {k: rng.randint(10, 99) for k in keys}
    facts = [f"{k}={v}" for k, v in mapping.items()]
    rng.shuffle(facts)
    q = rng.choice(keys)
    return f"task associative recall\nfacts {' '.join(facts)}\nquery {q}\nanswer ", str(mapping[q]), {"n_keys": n, "query": q}


def _state(rng: random.Random, *, ood: bool):
    items = ["red", "blue", "green", "gold", "white"] if not ood else ["red", "blue", "green", "gold", "white", "black", "silver", "orange"]
    locs = ["box", "desk", "shelf", "bag", "tray", "room"] if not ood else ["box", "desk", "shelf", "bag", "tray", "room", "vault", "yard", "cart", "case"]
    n_events = 24 if ood else 12
    initial = {x: rng.choice(locs) for x in items}
    state = dict(initial)
    events = []
    for _ in range(n_events):
        obj = rng.choice(items); dst = rng.choice(locs)
        state[obj] = dst; events.append((obj, dst))
    q = rng.choice(items)
    initial_txt = " ".join(f"{k}={initial[k]}" for k in items)
    event_txt = " ; ".join(f"move {a} to {b}" for a, b in events)
    prompt = f"task state tracking\ninitial {initial_txt}\n{event_txt}\nwhere {q}\nanswer "
    return prompt, state[q], {"n_events": n_events, "query": q, "initial": initial, "events": events, "verified_final": state[q]}


def _arith(rng: random.Random, *, ood: bool):
    if not ood:
        a, b, c = [rng.randint(10, 99) for _ in range(3)]
        if rng.random() < 0.5:
            expr, ans, pattern = f"{a}+{b}-{c}", a + b - c, "+-"
        else:
            expr, ans, pattern = f"{a}+{b}+{c}", a + b + c, "++"
    else:
        a, b, c, d = [rng.randint(100, 999) for _ in range(4)]
        if rng.random() < 0.5:
            expr, ans, pattern = f"{a}+{b}-{c}+{d}", a + b - c + d, "+-+"
        else:
            expr, ans, pattern = f"{a}+{b}+{c}-{d}", a + b + c - d, "++-"
    return f"task arithmetic\ncompute {expr}\nanswer ", str(ans), {"expr": expr, "pattern": pattern}


def _rule(rng: random.Random, *, ood: bool):
    if not ood:
        start = rng.randint(1, 30); mul = rng.choice([2, 3]); add = rng.randint(1, 10)
    else:
        start = rng.randint(1, 20); mul = rng.choice([4, 5]); add = rng.randint(11, 20)
    seq = [start]
    for _ in range(4):
        seq.append(seq[-1] * mul + add)
    ans = seq[-1] * mul + add
    return f"task rule induction\nsequence {' '.join(map(str, seq))}\nnext\nanswer ", str(ans), {"mul": mul, "add": add, "sequence": seq}


def _has_path(edges: Sequence[tuple[str, str]], src: str, dst: str) -> bool:
    adj: Dict[str, List[str]] = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
    seen = {src}; stack = [src]
    while stack:
        a = stack.pop()
        if a == dst:
            return True
        for b in adj.get(a, []):
            if b not in seen:
                seen.add(b); stack.append(b)
    return dst in seen


def _graph_for_label(rng: random.Random, *, ood: bool, reachable: bool):
    n_nodes = 14 if ood else 10
    nodes = _names(n_nodes, "n")
    edges: set[tuple[str, str]] = set()
    target_edges = 20 if ood else 14
    if reachable:
        src, dst = nodes[0], nodes[-1]
        mids = nodes[1:-1].copy(); rng.shuffle(mids)
        chain_len = 7 if ood else 5
        chain = [src] + mids[:chain_len] + [dst]
        edges.update(zip(chain[:-1], chain[1:]))
        while len(edges) < target_edges:
            edges.add(tuple(rng.sample(nodes, 2)))
        answer = "yes"
    else:
        cut = len(nodes) // 2
        left, right = nodes[:cut], nodes[cut:]
        src, dst = left[0], right[-1]
        while len(edges) < target_edges:
            comp = left if rng.random() < 0.5 else right
            edges.add(tuple(rng.sample(comp, 2)))
        answer = "no"
    edge_list = sorted(edges); rng.shuffle(edge_list)
    verified = _has_path(edge_list, src, dst)
    assert verified == reachable
    edge_txt = " ".join(f"{a}->{b}" for a, b in edge_list)
    return f"task graph\nedges {edge_txt}\nreachable {src} {dst}\nanswer ", answer, {"n_nodes": n_nodes, "src": src, "dst": dst, "edges": edge_list, "verified_reachable": verified}


def _reverse(rng: random.Random, *, ood: bool):
    n = 28 if ood else 14
    chars = "abcdefgxyz"
    s = "".join(rng.choice(chars) for _ in range(n))
    return f"task reverse\ninput {s}\nanswer ", s[::-1], {"length": n, "input": s}


GENERATORS: Mapping[str, Callable[..., tuple[str, str, dict]]] = {"assoc": _assoc, "state": _state, "arith": _arith, "rule": _rule, "reverse": _reverse}


def _seed_for(base: int, task: str) -> int:
    tag = int(hashlib.sha256(task.encode()).hexdigest()[:8], 16)
    return (base ^ tag) & 0xFFFFFFFF


def _build_suite_raw(name: str, exclude_prompt_hashes: set[str] | None = None) -> List[BenchCase]:
    if name not in {"dev-core", "dev-ood", "holdout-core", "holdout-ood"}:
        raise ValueError(name)
    is_holdout = name.startswith("holdout"); ood = name.endswith("ood")
    n = HOLDOUT_N_PER_TASK if is_holdout else DEV_N_PER_TASK
    base_seed = (HOLDOUT_SEEDS if is_holdout else DEV_SEEDS)["ood" if ood else "core"]
    rows: List[BenchCase] = []; excluded = exclude_prompt_hashes or set()
    for task in TASKS:
        rng = random.Random(_seed_for(base_seed, task)); generated = []; seen_prompts = set()
        if task == "graph":
            labels = [True] * (n // 2) + [False] * (n // 2)
            random.Random(_seed_for(base_seed + 17, task)).shuffle(labels)
            for y in labels:
                for _attempt in range(10000):
                    row = _graph_for_label(rng, ood=ood, reachable=y)
                    ph = hashlib.sha256(row[0].encode()).hexdigest()
                    if row[0] not in seen_prompts and ph not in excluded:
                        seen_prompts.add(row[0]); generated.append(row); break
                else:
                    raise RuntimeError(f"could not generate unique graph prompt for {name}")
        else:
            while len(generated) < n:
                row = GENERATORS[task](rng, ood=ood); ph = hashlib.sha256(row[0].encode()).hexdigest()
                if row[0] in seen_prompts or ph in excluded:
                    continue
                seen_prompts.add(row[0]); generated.append(row)
        for i, (prompt, answer, meta) in enumerate(generated):
            if not prompt.endswith("answer "):
                raise AssertionError("benchmark prompt must end at answer marker")
            rows.append(BenchCase(VERSION, name, task, i, prompt, answer, meta))
    audit_suite(rows)
    return rows


def build_suite(name: str) -> List[BenchCase]:
    if name == "holdout-core":
        dev = _build_suite_raw("dev-core")
        return _build_suite_raw(name, {x.prompt_hash for x in dev})
    if name == "holdout-ood":
        dev = _build_suite_raw("dev-ood")
        return _build_suite_raw(name, {x.prompt_hash for x in dev})
    return _build_suite_raw(name)


def audit_suite(rows: Sequence[BenchCase]) -> None:
    if not rows:
        raise AssertionError("empty suite")
    ids = [x.case_id for x in rows]; prompts = [x.prompt_hash for x in rows]
    if len(ids) != len(set(ids)) or len(prompts) != len(set(prompts)):
        raise AssertionError("duplicate benchmark cases/prompts")
    by_task: Dict[str, List[BenchCase]] = {t: [] for t in TASKS}
    for x in rows:
        by_task[x.task].append(x)
    if len({len(v) for v in by_task.values()}) != 1:
        raise AssertionError("task sizes differ")
    graph = by_task["graph"]
    if sum(x.answer == "yes" for x in graph) != sum(x.answer == "no" for x in graph):
        raise AssertionError("graph is not exactly balanced")
    for x in by_task["state"]:
        if "initial " not in x.prompt or x.metadata["verified_final"] != x.answer:
            raise AssertionError("invalid state case")
    for x in graph:
        if bool(x.metadata["verified_reachable"]) != (x.answer == "yes"):
            raise AssertionError("graph verifier disagrees")


def suite_digest(rows: Sequence[BenchCase]) -> str:
    h = hashlib.sha256()
    for x in rows:
        h.update(x.case_id.encode()); h.update(b"\n")
    return h.hexdigest()


def normalize_answer(task: str, text: str) -> str:
    s = text.strip()
    if task in {"graph", "state"}:
        return s.lower()
    if task in {"assoc", "arith", "rule"}:
        return str(int(s)) if re.fullmatch(r"[+-]?\d+", s) else s
    if task == "reverse":
        return s
    raise ValueError(task)


def majority_baseline(rows: Sequence[BenchCase]) -> Dict[str, float]:
    out = {}
    for task in TASKS:
        ans = [normalize_answer(task, x.answer) for x in rows if x.task == task]; counts = {}
        for a in ans: counts[a] = counts.get(a, 0) + 1
        out[task] = max(counts.values()) / len(ans)
    return out


def score_predictions(rows: Sequence[BenchCase], predictions: Mapping[str, str]) -> dict:
    baselines = majority_baseline(rows); per_task = {}
    for task in TASKS:
        task_rows = [x for x in rows if x.task == task]; correct = 0; missing = 0
        for x in task_rows:
            if x.case_id not in predictions:
                missing += 1; continue
            correct += int(normalize_answer(task, predictions[x.case_id]) == normalize_answer(task, x.answer))
        raw = correct / len(task_rows); b = baselines[task]
        per_task[task] = {"raw_exact": raw, "majority_baseline": b, "adjusted_exact": (raw - b) / max(1e-12, 1.0 - b), "correct": correct, "n": len(task_rows), "missing": missing}
    return {"version": VERSION, "suite": rows[0].suite, "suite_digest": suite_digest(rows), "raw_solve_macro": sum(x["raw_exact"] for x in per_task.values()) / len(TASKS), "normalized_solve_macro": sum(x["adjusted_exact"] for x in per_task.values()) / len(TASKS), "per_task": per_task}


def prompt_hash_from_training_text(text: str) -> str:
    if "answer " not in text:
        raise ValueError("training example lacks answer marker")
    prefix = text.rsplit("answer ", 1)[0] + "answer "
    return hashlib.sha256(prefix.encode()).hexdigest()


def contamination_report(rows: Sequence[BenchCase], train_prompt_hashes: Iterable[str]) -> dict:
    train = {x.strip() for x in train_prompt_hashes if x.strip()}; overlaps = [x for x in rows if x.prompt_hash in train]
    return {"suite": rows[0].suite, "n_cases": len(rows), "n_exact_prompt_overlaps": len(overlaps), "overlap_case_ids": [x.case_id for x in overlaps[:50]], "certified_clean": len(overlaps) == 0}


def manifest() -> dict:
    suites = {}; all_rows = {}
    for name in ("dev-core", "dev-ood", "holdout-core", "holdout-ood"):
        rows = build_suite(name); all_rows[name] = rows
        suites[name] = {"n": len(rows), "n_per_task": len(rows) // len(TASKS), "digest": suite_digest(rows), "majority_baseline": majority_baseline(rows)}
    names = list(all_rows); cross = {}
    for i, a in enumerate(names):
        ha = {x.prompt_hash for x in all_rows[a]}
        for b in names[i+1:]:
            overlap = len(ha & {x.prompt_hash for x in all_rows[b]}); cross[f"{a}__{b}"] = overlap
            if overlap:
                raise AssertionError(f"prompt overlap between {a} and {b}: {overlap}")
    return {"version": VERSION, "tasks": list(TASKS), "suites": suites, "cross_suite_prompt_overlap": cross}


if __name__ == "__main__":
    print(json.dumps(manifest(), indent=2, sort_keys=True))
