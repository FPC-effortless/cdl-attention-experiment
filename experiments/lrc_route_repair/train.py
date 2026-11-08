"""Training-only dense supervision; deployment uses model.py's bounded index."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from lrc_vb001_comparison.source import study as vb
from lrc_vb001_comparison.model import SentenceLRC
from lrc_vb001_comparison.train import update as original_update
from .model import ContentAddressedLRC
from .key_learning_probe import spherical_kmeans


ARMS = ("repair", "repair_shuffled", "original_lrc", "memnet")


def build(arm):
    if arm in ("repair", "repair_shuffled"):
        return ContentAddressedLRC()
    if arm == "original_lrc":
        return SentenceLRC(29)
    if arm == "memnet":
        return vb.MemoryNetwork(51)
    raise ValueError(arm)


def teacher_keys(model, batch, supports):
    values, keys = model.router.write(batch["records"])
    state, first = model.router.query_first(batch["query"])
    rows = torch.arange(len(values))
    second = model.router.query_second(state, values[rows, supports[:, 0]])
    reads = torch.stack((first, second), 1)
    return values, keys, reads


def fit(arm, seed, train, out, *, steps=2000, key_steps=1600, batch_size=32, development=False):
    if not 0 < key_steps < steps:
        raise ValueError("the recipe requires both key-learning and distillation phases")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out / f"{arm}_{seed}.pt"
    log_path = out / f"{arm}_{seed}.jsonl"
    if checkpoint_path.exists() or log_path.exists():
        raise RuntimeError("refusing to overwrite an experiment")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = build(arm)
    generator = torch.Generator().manual_seed(100000 + seed)
    shuffle_generator = torch.Generator().manual_seed(200000 + seed)
    order_hash = hashlib.sha256()
    is_repair = arm.startswith("repair")
    if is_repair:
        active_key_parameters = list(model.router.parameters())
        key_optimizer = torch.optim.AdamW(active_key_parameters, lr=0.003, weight_decay=0)
        answer_optimizer = torch.optim.AdamW(model.answer.parameters(), lr=0.003, weight_decay=0)
        stage1_optimizer_state = None
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0)
    codebook_receipt = None
    started = time.perf_counter()
    training_forwards = 0
    with log_path.open("x") as log:
        for step in range(steps):
            idx = torch.randint(len(train["answer"]), (batch_size,), generator=generator)
            order_hash.update(idx.numpy().tobytes())
            batch = vb.take(train, idx)
            model.train()
            if is_repair:
                if step == key_steps:
                    phase_started = time.perf_counter()
                    strings = torch.unique(train["records"].flatten(0, 1), dim=0)
                    with torch.no_grad():
                        _, points = model.router.write(strings)
                        inertia, centers = spherical_kmeans(points, count=16, seed=seed + 20000,
                                                            restarts=5, iterations=100)
                        model.router.codebook.copy_(centers)
                    codebook_receipt = dict(unique_training_strings=len(strings), inertia=inertia,
                                            restarts=5, iterations_max=100,
                                            seconds=time.perf_counter() - phase_started)
                    stage1_optimizer_state = key_optimizer.state_dict()
                    for parameter in list(model.router.encoder.parameters()) + list(model.router.writer.parameters()):
                        parameter.requires_grad_(False)
                        parameter.grad = None
                    active_key_parameters = [p for p in model.router.parameters() if p.requires_grad]
                    key_optimizer = torch.optim.AdamW(active_key_parameters,
                                                       lr=0.002, weight_decay=0)
                    for group in answer_optimizer.param_groups:
                        group["lr"] = 0.002
                supports = batch["support"]
                if arm == "repair_shuffled":
                    supports = supports[torch.randperm(len(supports), generator=shuffle_generator)]
                values, keys, reads = teacher_keys(model, batch, supports)
                similarity = torch.einsum("bhd,bnd->bhn", reads, keys)
                contrastive = F.cross_entropy((similarity / 0.08).flatten(0, 1), supports.flatten())
                if step < key_steps:
                    positive = 1 - similarity.gather(-1, supports.unsqueeze(-1)).mean()
                    key_loss = contrastive + positive
                else:
                    target_codes = (keys @ model.router.codebook.T).argmax(-1).gather(1, supports)
                    key_loss = F.cross_entropy((reads @ model.router.codebook.T / 0.08).flatten(0, 1),
                                               target_codes.flatten()) + 0.25 * contrastive
                # Detached sentence values prevent answer loss changing key learning.
                gold_final = values[torch.arange(len(values)), supports[:, 1]].detach()
                logits = model.answer(gold_final)
                answer_loss = F.cross_entropy(logits, batch["answer"])
                key_optimizer.zero_grad(set_to_none=True)
                key_loss.backward()
                key_norm = nn.utils.clip_grad_norm_(active_key_parameters, 1)
                key_optimizer.step()
                answer_optimizer.zero_grad(set_to_none=True)
                answer_loss.backward()
                answer_norm = nn.utils.clip_grad_norm_(model.answer.parameters(), 1)
                answer_optimizer.step()
                if not torch.isfinite(key_norm + answer_norm):
                    raise RuntimeError("nonfinite gradient")
                record = dict(step=step + 1, key_loss=float(key_loss.detach()),
                              answer_loss=float(answer_loss.detach()),
                              key_gradient_norm=float(key_norm), answer_gradient_norm=float(answer_norm))
                training_forwards += 1  # one encoded training batch; two separate losses
            elif arm == "original_lrc":
                record = original_update(model, optimizer, batch, "lrc_agreement", 2300000 + seed * 10000 + step)
                record["step"] = step + 1
                training_forwards += 2
            else:
                optimizer.zero_grad(set_to_none=True)
                loss, answer_loss, aux = vb.objective(model, batch, 1.0)
                loss.backward()
                norm = nn.utils.clip_grad_norm_(model.parameters(), 1)
                if not torch.isfinite(norm):
                    raise RuntimeError("nonfinite gradient")
                optimizer.step()
                record = dict(step=step + 1, answer_loss=float(answer_loss.detach()),
                              support_loss=float(aux.detach()), gradient_norm=float(norm))
                training_forwards += 1
            log.write(json.dumps(record) + "\n")
            if (step + 1) % 400 == 0:
                log.flush()
                print(json.dumps(dict(arm=arm, seed=seed, **record)), flush=True)
    seconds = time.perf_counter() - started
    receipt = dict(arm=arm, seed=seed, steps=steps, batch=batch_size, examples_seen=steps * batch_size,
                   parameters=sum(p.numel() for p in model.parameters()), fit_seconds=seconds,
                   training_forwards=training_forwards,
                   optimizer_step_calls=steps * (2 if is_repair else 1),
                   stage1_trainable_parameters=sum(p.numel() for p in model.parameters()),
                   stage2_trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                   codebook=codebook_receipt, minibatch_sha256=order_hash.hexdigest(),
                   log_sha256=hashlib.sha256(log_path.read_bytes()).hexdigest(),
                   train_sha256=vb.digest_data(train), development_only=development,
                   torch_version=str(torch.__version__), threads=1,
                   finished_unix=time.time())
    state = dict(model=model.state_dict(), receipt=receipt, rng=torch.get_rng_state(),
                 minibatch_rng=generator.get_state(), shuffle_rng=shuffle_generator.get_state())
    if is_repair:
        state.update(key_optimizer_stage1=stage1_optimizer_state,
                     key_optimizer=key_optimizer.state_dict(), answer_optimizer=answer_optimizer.state_dict())
    else:
        state["optimizer"] = optimizer.state_dict()
    torch.save(state, checkpoint_path)
    receipt["checkpoint_sha256"] = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    (out / f"{arm}_{seed}.receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    model.eval()
    return model, receipt
