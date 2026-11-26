"""Prospectively budgeted matched training; held panels are never loaded here."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from .data import UNKNOWN, digest
from .engine import Resolution, evidence_tensors, execute, pad_texts, predicted_operation, strict_read
from .model import FusionModel


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_digest(model):
    output = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        output.update(name.encode())
        output.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return output.hexdigest()


def route_loss(model, features, worlds, resolutions, teacher):
    batch = len(worlds)
    entities = torch.tensor([w["query_entity"] for w in worlds])
    relations = torch.tensor([0 if w["query_plan"] == 0 else 1 for w in worlds])
    all_e = torch.arange(16).repeat_interleave(2)[None].expand(batch, -1)
    all_r = torch.arange(2).repeat(16)[None].expand(batch, -1)
    buckets, query = model.route_logits(features)
    first = F.cross_entropy(buckets, model.bucket(entities, relations))
    first = first + F.cross_entropy(model.candidate_scores(query, all_e, all_r), entities * 2 + relations)
    previous = torch.tensor([w["evidence_values"][0] if teacher else res.values[0]
                             for w, res in zip(worlds, resolutions)])
    mask = torch.tensor([w["query_plan"] == 2 for w in worlds]) & (previous != UNKNOWN)
    if bool(mask.any()):
        buckets2, query2 = model.route_logits(features[mask], previous_value=previous[mask], hop=1)
        target = previous[mask]
        second = F.cross_entropy(buckets2, target % 16)
        second = second + F.cross_entropy(model.candidate_scores(query2, all_e[mask], all_r[mask]), target * 2)
        return (first + second) / 2
    return first


def batch_loss(model, families, *, teacher=False, lm_tokens=None):
    worlds = [world for family in families for world in family["worlds"]]
    batch = len(worlds)
    state = model.init_state(batch)
    ledgers = [{} for _ in worlds]
    semantic_loss = torch.zeros(())
    write_correct = 0
    input_chars = 0
    physical_tokens = 0
    for position in range(len(worlds[0]["events"])):
        events = [w["events"][position] for w in worlds]
        texts = [e["text"] + "\n" for e in events]
        tokens, lengths = model.encode_texts(texts)
        input_chars += int(lengths.sum())
        physical_tokens += tokens.numel()
        features, semantics, state = model.encode_event(tokens, lengths, state)
        gold = {key: torch.tensor([e[key] for e in events]) for key in ("operation", "entity", "relation", "value")}
        event_loss = F.cross_entropy(semantics["operation"], gold["operation"])
        changed = gold["operation"] != 0
        assigned = gold["operation"] == 1
        if bool(changed.any()):
            event_loss = event_loss + F.cross_entropy(semantics["entity"][changed], gold["entity"][changed])
            event_loss = event_loss + 0.5 * F.cross_entropy(semantics["relation"][changed], gold["relation"][changed])
        if bool(assigned.any()):
            event_loss = event_loss + F.cross_entropy(semantics["value"][assigned], gold["value"][assigned])
        semantic_loss = semantic_loss + event_loss
        for i, event in enumerate(events):
            predicted = predicted_operation(semantics, i)
            expected = tuple(event[k] for k in ("operation", "entity", "relation", "value"))
            exact = predicted[0] == expected[0] and (expected[0] == 0 or
                    (predicted[1:3] == expected[1:3] and (expected[0] != 1 or predicted[3] == expected[3])))
            write_correct += int(exact)
            execute(ledgers[i], expected if teacher else predicted)
    semantic_loss = semantic_loss / len(worlds[0]["events"])
    tokens, lengths = model.encode_texts([w["query"] + "\n" for w in worlds])
    input_chars += int(lengths.sum())
    physical_tokens += tokens.numel()
    features, semantics, state = model.encode_event(tokens, lengths, state)
    entity_targets = torch.tensor([w["query_entity"] for w in worlds])
    plan_targets = torch.tensor([w["query_plan"] for w in worlds])
    semantic_loss = semantic_loss + F.cross_entropy(semantics["query_entity"], entity_targets)
    semantic_loss = semantic_loss + 0.5 * F.cross_entropy(semantics["query_plan"], plan_targets)
    entities = entity_targets if teacher else semantics["query_entity"].argmax(-1)
    plans = plan_targets if teacher else semantics["query_plan"].argmax(-1)
    resolutions = [strict_read(ledger, int(e), int(p)) for ledger, e, p in zip(ledgers, entities, plans)]
    routing = route_loss(model, features, worlds, resolutions, teacher)
    prefix = "\nAnswer: "
    inputs = [prefix + w["answer_text"][:-1] for w in worlds]
    tokens, lengths = model.encode_texts(inputs)
    decode_input_chars = int(lengths.sum())
    physical_tokens += tokens.numel()
    evidence, mask = evidence_tensors(resolutions)
    logits, _ = model.decode(tokens, state, evidence, mask, lengths=lengths)
    labels = torch.full(tokens.shape, -100, dtype=torch.long)
    for i, world in enumerate(worlds):
        ids = torch.tensor([model.stoi[c] for c in world["answer_text"]])
        labels[i, len(prefix) - 1:len(prefix) - 1 + len(ids)] = ids
    answer_loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100)
    first_logp = logits[:, len(prefix) - 1].log_softmax(-1)
    pair_losses = []
    for i, fam in enumerate(families):
        left, right = i * 2, i * 2 + 1
        a, b = worlds[left]["answer_text"][0], worlds[right]["answer_text"][0]
        if a != b:
            ia, ib = model.stoi[a], model.stoi[b]
            pair_losses.append(F.relu(0.5 - first_logp[left, ia] + first_logp[left, ib]))
            pair_losses.append(F.relu(0.5 - first_logp[right, ib] + first_logp[right, ia]))
        else:
            pa, pb = first_logp[left].exp(), first_logp[right].exp()
            pair_losses.append(((pa - pb) * (first_logp[left] - first_logp[right])).sum() / 2)
    paired = torch.stack(pair_losses).mean()
    loss = semantic_loss + 0.5 * routing + 2.0 * answer_loss + 0.2 * paired
    rehearsal = torch.zeros(())
    if lm_tokens is not None:
        physical_tokens += lm_tokens[:, :-1].numel()
        lm_logits, _ = model.forward_lm(lm_tokens[:, :-1])
        rehearsal = F.cross_entropy(lm_logits.flatten(0, 1), lm_tokens[:, 1:].flatten())
        loss = loss + 0.2 * rehearsal
    return loss, {"semantic_loss": float(semantic_loss.detach()), "answer_loss": float(answer_loss.detach()),
                  "route_loss": float(routing.detach()), "pair_loss": float(paired.detach()),
                  "lm_loss": float(rehearsal.detach()), "write_exact": write_correct,
                  "write_total": batch * len(worlds[0]["events"]),
                  "input_chars": input_chars, "output_targets": int((labels != -100).sum()),
                  "decode_input_chars": decode_input_chars, "physical_token_positions": physical_tokens,
                  "rehearsal_targets": 0 if lm_tokens is None else int(lm_tokens[:, 1:].numel())}


def run(args):
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    rng = random.Random(90017 + args.seed)
    model = FusionModel.from_native(args.native, args.seed, memory_enabled=args.arm == "memory")
    corpus = json.loads(args.corpus.read_text())
    # Immediately discard held content; never score or select against it.
    families = corpus["train"]
    del corpus
    lm = np.memmap(args.shakespeare, dtype="<u2", mode="r")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "final.pt").exists():
        raise FileExistsError("refusing to overwrite completed run")
    source = {str(p.relative_to(Path(__file__).parents[1])): file_sha(p)
              for p in sorted((Path(__file__).parent).rglob("*.py"))}
    identity = {"arm": args.arm, "seed": args.seed, "steps": args.steps, "batch_worlds": args.pairs * 2,
                "teacher_steps": args.teacher_steps, "threads": args.threads,
                "corpus_sha256": file_sha(args.corpus), "train_data_sha256": digest(families),
                "native_sha256": file_sha(args.native), "shakespeare_train_sha256": file_sha(args.shakespeare),
                "parameters": model.parameter_summary(), "initial_model_identity": model.model_identity(),
                "initial_tensor_sha256": tensor_digest(model),
                "source_sha256": source, "optimizer": "AdamW", "weight_decay": 0.01,
                "native_lr": args.native_lr, "interface_lr": args.lr, "gradient_clip": 1.0,
                "bptt": "entire five-event history and answer, no intermediate state detach",
                "teacher_policy": "gold typed writes/query for initial steps, then entirely predicted typed execution",
                "training_reader": "strict", "hybrid_supervision": "16 bucket classes and 32 identity key comparisons per hop",
                "checkpoint_selection": "fixed final update; no held-data selection"}
    (args.output / "IDENTITY.json").write_text(json.dumps(identity, indent=2) + "\n")
    native_params = [p for name, p in model.named_parameters() if name.startswith("native.")]
    new_params = [p for name, p in model.named_parameters() if not name.startswith("native.")]
    optimizer = torch.optim.AdamW([{"params": native_params, "lr": args.native_lr},
                                  {"params": new_params, "lr": args.lr}], weight_decay=0.01)
    base_lrs = [args.native_lr, args.lr]
    started, cpu_start = time.perf_counter(), time.process_time()
    counters = {"input_chars": 0, "output_targets": 0, "rehearsal_targets": 0,
                "decode_input_chars": 0, "physical_token_positions": 0}
    batch_trace = hashlib.sha256()
    model.train()
    for step in range(1, args.steps + 1):
        selection = [rng.randrange(len(families)) for _ in range(args.pairs)]
        batch_trace.update(json.dumps(selection).encode())
        batch = [families[i] for i in selection]
        rehearsal = None
        if step % 4 == 0:
            starts = [rng.randrange(len(lm) - 129) for _ in range(4)]
            rehearsal = torch.tensor(np.stack([np.array(lm[start:start + 129], dtype=np.int64) for start in starts]))
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base_lr * (0.2 + 0.8 * (1 + math.cos(math.pi * (step - 1) / args.steps)) / 2)
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = batch_loss(model, batch, teacher=step <= args.teacher_steps, lm_tokens=rehearsal)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite training loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(norm):
            raise FloatingPointError("nonfinite gradient")
        optimizer.step()
        for key in counters:
            counters[key] += metrics[key]
        row = {"step": step, "loss": float(loss.detach()), **metrics,
               "gradient_norm": float(norm), "teacher": step <= args.teacher_steps,
               "wall_seconds": time.perf_counter() - started, "cpu_seconds": time.process_time() - cpu_start}
        with (args.output / "train.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        if step == 1 or step % 50 == 0 or step == args.steps:
            print(json.dumps(row), flush=True)
        if row["cpu_seconds"] > args.cpu_cap:
            torch.save(model.checkpoint_payload(step=step, stopped="cpu_cap", **counters), args.output / "incomplete.pt")
            raise RuntimeError("prospective CPU cap exceeded; incomplete run retained")
    elapsed = {"wall_seconds": time.perf_counter() - started, "cpu_seconds": time.process_time() - cpu_start}
    model.eval()
    torch.save(model.checkpoint_payload(step=args.steps, seed=args.seed, arm=args.arm,
                                       batch_trace_sha256=batch_trace.hexdigest(), **elapsed, **counters), args.output / "final.pt")
    torch.save({"optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
                "python_rng": rng.getstate(), "step": args.steps}, args.output / "optimizer.pt")
    result = {**identity, **elapsed, **counters, "batch_trace_sha256": batch_trace.hexdigest(),
              "final_model_identity": model.model_identity(), "checkpoint_sha256": file_sha(args.output / "final.pt"),
              "status": "completed"}
    (args.output / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--shakespeare", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("memory", "no_memory"), required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--teacher-steps", type=int, default=200)
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--cpu-cap", type=float, default=1800)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--native-lr", type=float, default=0.00015)
    run(parser.parse_args())
