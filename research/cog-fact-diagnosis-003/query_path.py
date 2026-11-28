"""Inspect the native output and local routing at supervised positions in a question."""
from collections import defaultdict
from pathlib import Path
import sys

from trace_fault import cases, ids, json, loadstudy, mean, np, torch


@torch.inference_mode()
def evaluate(model, domain):
    worlds = list(cases(192, 957427, True, domain, independent=True))
    qs = ("#a00=", "#b00=") if domain == "symbol" else tuple(
        f"What digit does the log give for {name}?" for name in ("Ada", "Ben"))
    stage = {"query_key": 1, "operation_1": 2, "operation_2": 3, "delimiter": 4} if domain == "symbol" else {"name_end": -2, "delimiter": -1}
    results = defaultdict(list)
    for start in range(0, len(worlds), 16):
        group = worlds[start:start + 16]
        x = ids([w[i] for w in group for i in (0, 1)])
        _, state, _ = model(x, features=True)
        scores, route = [], []
        for q in qs:
            query = ids([q]*len(x))
            z, _, _ = model(query, state=state, features=True)
            _, _, aux = model.parent.local(query, model.parent.core.emb, state[1])
            scores.append(z)
            route.append(torch.stack([aux[0].argmax(-1), aux[1].argmax(-1)], -1))
        for label, idx in stage.items():
            results[label].append(torch.stack([z[:, idx, :] for z in scores], 1).numpy())
            results[label + "_route"].append(torch.stack([r[:, idx, :] for r in route], 1).numpy())
    truth = np.array([[[a,b],[b,a]] for _,_,a,b in worlds]).reshape(-1,2)
    report = {}
    for label in stage:
        scores = np.concatenate(results[label])
        r = np.concatenate(results[label + "_route"])
        pred = scores.argmax(-1) - 48
        digits = scores[:, :, 48:52]
        right = np.take_along_axis(digits, truth[:,:,None], axis=-1)[:,:,0]
        other = np.take_along_axis(digits, truth[:,::-1,None], axis=-1)[:,:,0]
        # Signed query-dependent evidence for the correct assignment in each history.
        binding_margin = (right-other).mean(1)
        report[label] = {"correct":mean(pred==truth),"four_digit_correct":mean(digits.argmax(-1)==truth),
            "grid_correct":int((pred==truth).reshape(-1,4).all(1).sum()),
            "answer_changes_on_query":mean(pred[:,0]!=pred[:,1]),
            "mean_binding_margin":mean(binding_margin),
            "positive_binding_margin":mean(binding_margin>0),
            "role_by_query":r[0,:,0].tolist(),"address_by_query":r[0,:,1].tolist(),
            "route_differs_on_query":mean((r[:,0]!=r[:,1]).any(-1))}
    return report


if __name__ == "__main__":
    torch.set_num_threads(1)
    model, arm, scale, seed, sha = loadstudy(sys.argv[1])
    out = {domain:evaluate(model,domain) for domain in ("symbol","english")}
    Path(sys.argv[2]).write_text(json.dumps(dict(seed=seed, checkpoint_sha256=sha, **out), indent=2))
