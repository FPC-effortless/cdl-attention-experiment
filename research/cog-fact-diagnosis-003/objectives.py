"""Compare local training gradients at frozen checkpoint weights, without optimizer updates."""
import sys
from collections import defaultdict
from pathlib import Path

from trace_fault import F, json, load_corpus, loadstudy, np, torch
from repair_objective import repair_batch, repair_loss


def gradients(loss, parameters):
    values = torch.autograd.grad(loss, parameters, allow_unused=True)
    return [g.detach().flatten() if g is not None else None for g in values]


def evaluate(model, corpus):
    model.train()  # Match the original differentiable hard-routing implementation.
    names, parameters = zip(*model.named_parameters())
    rng = np.random.default_rng(305173)
    groups = {"all": range(len(names)),
              "local": [i for i, n in enumerate(names) if ".local." in n],
              "blocks": [i for i, n in enumerate(names) if ".blocks." in n],
              "embedding": [i for i, n in enumerate(names) if n.endswith("emb.weight")]}
    if any(not indices for indices in groups.values()):
        raise ValueError("Missing parameter group")
    results = defaultdict(list)
    for step in range(32):
        xy = torch.from_numpy(corpus.batch("dev", rng, 4, 128))
        language_logits = model(xy[:, :-1])
        lang = gradients(F.cross_entropy(language_logits.flatten(0, 1), xy[:, 1:].flatten()), parameters)
        batch = repair_batch(384716, step, batch_size=8, balanced=True)
        task = gradients(repair_loss(model(batch.ids), batch, "trace_pair"), parameters)
        for group, indices in groups.items():
            dot = sum(float(task[i] @ lang[i]) for i in indices if task[i] is not None and lang[i] is not None)
            target_norm = sum(float(task[i] @ task[i]) for i in indices if task[i] is not None)
            lm_norm = sum(float(lang[i] @ lang[i]) for i in indices if lang[i] is not None)
            if target_norm and lm_norm:
                results[group].append(dot / (target_norm * lm_norm)**0.5)
    return {k: {"batches": len(v), "mean_cosine": float(np.mean(v)), "negative_batches": sum(t < 0 for t in v),
                "standard_error": float(np.std(v, ddof=1)/np.sqrt(len(v))), "cosines": v} for k, v in results.items()}


if __name__ == "__main__":
    torch.set_num_threads(1)
    m, arm, scale, seed, sha = loadstudy(sys.argv[1])
    results = evaluate(m, load_corpus(Path(sys.argv[3])))
    Path(sys.argv[2]).write_text(json.dumps(dict(checkpoint_sha256=sha, seed=seed, groups=results), indent=2))
