"""Locate query identity and requested values with independent distractor digits."""
import sys
from collections import defaultdict
from pathlib import Path

from trace_fault import cases, get_state, ids, json, loadstudy, mean, np, probe, torch


def fit_decode(features, labels, split):
    return probe(features[:split], features[split:], labels[:split], labels[split:])


@torch.inference_mode()
def evaluate(model, native, domain="symbol"):
    train = list(cases(384, 861359, False, domain, independent=True))
    test = list(cases(192, 957427, True, domain, independent=True))
    questions = ("#a00=", "#b00=") if domain == "symbol" else tuple(
        f"What digit does the log give for {name}?" for name in ("Ada", "Ben"))
    worlds = train + test
    values = np.array([[[a, b], [b, a]] for _, _, a, b in worlds])
    states, logits = [], []
    layers = defaultdict(list)
    for start in range(0, len(worlds), 16):
        group = worlds[start:start + 16]
        histories = [world[i] for world in group for i in (0, 1)]
        _, state, _ = model(ids(histories), features=True)
        if native:
            parts = ["fast", "protected", "background"]
            if len(state) == 3:
                parts.append("slow")
            states.append(torch.cat([get_state(state, p) for p in parts], 1).numpy())
        responses = []
        captures = defaultdict(list)
        def capture(i):
            def hook(module, inputs, outputs):
                captures[i].append(outputs[0][:, -1].clone())
            return hook
        hooks = [block.register_forward_hook(capture(i)) for i, block in enumerate(model.parent.core.blocks)] if native else []
        try:
            for question in questions:
                scores, _, hidden = model(ids([question] * len(histories)), state=state, features=True)
                responses.append(scores[:, -1].argmax(-1))
                captures["final"].append(hidden[:, -1])
        finally:
            for hook in hooks:
                hook.remove()
        for key, features in captures.items():
            layers[key].append(torch.stack(features, 1).numpy())
        logits.append(torch.stack(responses, 1).numpy() - 48)
    n = 2 * len(train)
    y = values.reshape(-1, 2)
    truth = y[n:]
    output = np.concatenate(logits)[n:]
    result = {"worlds": len(test), "head_correct": mean(output == truth),
              "correct_grids": int((output == truth).reshape(-1, 4).all(1).sum()),
              "answer_changes_on_query": mean(output[:, 0] != output[:, 1]), "layers": {}}
    def by_key(x, labels):
        return np.stack([fit_decode(x[:, key], labels[:, key], n) for key in (0, 1)], 1)
    if states:
        x = np.concatenate(states)
        x = np.broadcast_to(x[:, None], (len(x), 2, x.shape[-1]))
        pred = by_key(x, y)
        result["keyed_history"] = mean(pred == truth)
        # Shuffle entire training worlds. Preserve swaps and balance within a grid.
        rng = np.random.default_rng(468341)
        shuffled = values.copy()
        shuffled[:len(train)] = shuffled[rng.permutation(len(train))]
        result["shuffled_history_labels"] = mean(by_key(x, shuffled.reshape(-1, 2)) == truth)
    for key, parts in layers.items():
        x = np.concatenate(parts)
        flat = x.reshape(-1, x.shape[-1])
        q = np.tile(np.arange(2), len(x))
        qpred = fit_decode(flat, q, 2*n)
        pred = fit_decode(flat, y.ravel(), 2*n)
        result["layers"][str(key)] = {"query_identity": mean(qpred == q[2*n:]),
            "keyed_digit": mean(by_key(x, y) == truth),
            "pooled_digit": mean(pred == truth.ravel())}
    return result


if __name__ == "__main__":
    torch.set_num_threads(1)
    model, arm, scale, seed, sha = loadstudy(sys.argv[1])
    domain = sys.argv[3] if len(sys.argv) > 3 else "symbol"
    result = evaluate(model, arm in ("A", "C"), domain)
    result.update(checkpoint_sha256=sha, arm=arm, scale=scale, seed=seed)
    Path(sys.argv[2]).write_text(json.dumps(result, indent=2))
