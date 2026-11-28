"""Check trace causality, intervention direction and end-to-end backpropagation."""
import argparse
import json

import numpy as np
import torch

from oracle_panels import grammar_answer
from panels import NAMES
from repair_objective import balanced_english_world, REGIMES, intervention_loss, repair_batch, repair_loss, trace_targets


def verify(seed=19612, steps=128):
    torch.manual_seed(102)
    torch.set_num_threads(1)
    prefixes = {}
    observations = 0
    worlds = 0
    unchanged = 0
    for step in range(steps):
        b = repair_batch(seed, step, 16)
        assert torch.equal(b.ids, repair_batch(seed, step, 16).ids)
        for i in range(0, len(b.ids), 2):
            left, right = b.ids[i:i+2]
            changed = (left != right).nonzero().flatten()
            assert changed.numel() == 1
            first = changed.item()
            assert torch.equal(b.trace_targets[i, :first], b.trace_targets[i+1, :first])
            unchanged += int(b.final_targets[i] == b.final_targets[i+1])
            for j in (i, i+1):
                prompt = bytes(b.ids[j].tolist())
                events, answer = trace_targets(prompt)
                assert answer == b.final_targets[j] == grammar_answer(prompt)
                assert b.trace_targets[j, -1] == -100
                for pos, digit in events:
                    assert b.trace_targets[j,pos] == digit
                    history = prompt[:pos+1]
                    if history in prefixes:
                        assert prefixes[history] == digit
                    prefixes[history] = digit
                    observations += 1
            worlds += 1

    intervention_checks = {}
    generator = np.random.default_rng(seed + 731)
    for kind in ("fact", "query", "rumor"):
        for i in range(256):
            old, new, before, after = balanced_english_world(generator, i % 2, kind)
            for prompt, label in ((old,before),(new,after)):
                _, decoded = trace_targets(prompt)
                assert decoded == label == grammar_answer(prompt)
                history, query = prompt.rsplit(b". ", 1)
                assert all(history.count(name.encode()) == 2 for name in NAMES)
            old_history, old_query = old.rsplit(b". ",1)
            same_history = old_history == history
            assert same_history == (kind == "query")
            assert (old_query != query) == (kind == "query")
            assert (before == after) == (kind == "rumor")
            if kind != "query":
                assert sum(x != y for x,y in zip(old,new)) == 1
            first = next(j for j,(x,y) in enumerate(zip(old,new)) if x!=y)
            o, _ = trace_targets(old)
            n, _ = trace_targets(new)
            assert [(pos,y) for pos,y in o if pos<first] == [(pos,y) for pos,y in n if pos<first]
        intervention_checks[kind] = i + 1
    a, b = ord("1"), ord("3")
    targets = torch.tensor([a,b])
    logits = torch.zeros((2,288), requires_grad=True)
    initial = intervention_loss(logits, targets)
    initial.backward()
    changed_logits = logits.detach().clone()
    changed_logits[0,a] = changed_logits[1,b] = 2
    better = intervention_loss(changed_logits,targets)
    assert better < initial
    assert intervention_loss(changed_logits.flip(0),targets) > initial
    assert logits.grad[0,a] < 0 < logits.grad[1,a]
    invariant = intervention_loss(logits.detach(), torch.tensor([b,b]))
    assert abs(float(invariant)) < 1e-7
    assert intervention_loss(changed_logits, torch.tensor([b,b])) > invariant
    bch = repair_batch(seed,1,8)
    metrics = {}
    for name in REGIMES:
        predicted = torch.zeros((*bch.ids.shape,288), requires_grad=True)
        loss = repair_loss(predicted,bch,name)
        loss.backward()
        assert torch.isfinite(predicted.grad).all()
        mask = predicted.grad[:, :-1].abs().sum(-1) > 0
        assert torch.equal(mask, (bch.trace_targets[:, :-1]!=-100) if "trace" in name else torch.zeros_like(mask))
        metrics[name] = float(loss.detach())
    assert 0 < unchanged < worlds
    return {"paired_training_worlds":worlds,"balanced_english_pairs_by_intervention":intervention_checks,"checked_trace_positions":observations,
            "identical_prefix_label_keys":len(prefixes),"invariant_pairs":unchanged,
            "null_logit_loss":metrics,"change_loss_after_directional_update":float(better)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps",type=int,default=128)
    args=parser.parse_args()
    print(json.dumps(verify(steps=args.steps), indent=2))
