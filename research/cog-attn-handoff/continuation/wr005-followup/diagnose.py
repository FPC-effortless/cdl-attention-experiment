"""Compare proposal gradients to exhaustive gradients at pilot checkpoints."""
import json

from run import ROOT, action_mass, compatible, dataset, features, predict, proposals, shortlist
import numpy as np

PILOT = ROOT / "diagnostic-inputs"
COHORTS = ("familiar", "held_pair", "recombine")
MODES = ("top8", "conditioned", "unit16", "weighted16", "weighted32")


def terms(probs, actions, mask, n):
    mass = action_mass(probs, actions, n) * mask
    total = mass.sum(axis=1)
    posterior = np.divide(mass, total[:, None], out=np.zeros_like(mass), where=total[:, None] > 0)
    gradient = []
    for p, axis in zip(probs, (actions // n % n, actions % n, actions // (n * n))):
        delta = p * (total[:, None] > 0)
        np.add.at(delta, (np.arange(len(actions))[:, None], axis), -posterior)
        gradient.append(delta)
    return total, np.concatenate(gradient, axis=1)


def main():
    results = []
    n = 32
    for seed in (11, 29, 47):
        spec = json.loads((PILOT / f"data-{n}-{seed}.json").read_text())
        aliases = np.array(spec["aliases"])
        for origin in ("exhaustive", "conditioned"):
            with np.load(PILOT / f"{origin}-{n}-{seed}.npz", allow_pickle=False) as checkpoint:
                weights = [checkpoint[f"arr_{i}"] for i in range(3)]
            for offset, cohort in enumerate(COHORTS):
                raw, state, target, _ = dataset(seed + 7100 + offset, 256, n, aliases, cohort)
                tokens, _ = features(raw, spec["vocabulary"])
                probs = predict(weights, tokens)
                all_actions = np.broadcast_to(np.arange(2 * n * n), (len(raw), 2 * n * n))
                exact, reference = terms(probs, all_actions, compatible(all_actions, state, target), n)
                assert np.all(exact > 0)
                top = shortlist(probs, n)
                initial = compatible(top, state, target)
                changed = (state != target).any(axis=1)
                for mode in MODES:
                    for repeat in range(1 if mode == "top8" else 5):
                        rng = np.random.default_rng(seed * 1000 + offset * 10 + repeat)
                        actions, mask = (top, initial) if mode == "top8" else proposals(
                            rng, mode, top, initial, state, target)[:2]
                        estimated, gradient = terms(probs, actions, mask, n)
                        retained = (action_mass(probs, actions, n) * (mask > 0)).sum(1) / exact
                        norm = np.linalg.norm(reference, axis=1)
                        proposed_norm = np.linalg.norm(gradient, axis=1)
                        valid = (norm > 1e-12) & (proposed_norm > 1e-12)
                        cosine = np.divide((reference * gradient).sum(axis=1), norm * proposed_norm,
                                           out=np.zeros_like(norm), where=valid)
                        for label, group in (("changed", changed), ("noop", ~changed)):
                            selected = valid & group
                            error = np.linalg.norm(gradient - reference, axis=1) / np.maximum(norm, 1e-12)
                            results.append(dict(seed=seed, checkpoint=origin, cohort=cohort, mode=mode,
                                                repeat=repeat, transition=label, count=int(group.sum()),
                                                support=int((group & (estimated > 0)).sum()),
                                                retained_sum=float(retained[group].sum()),
                                                estimate_ratio_sum=float((estimated[group] / exact[group]).sum()),
                                                cosine_sum=float(cosine[selected].sum()),
                                                cosine_count=int(selected.sum()),
                                                norm_error_sum=float(error[group].sum())))
    (ROOT / "diagnostics.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
