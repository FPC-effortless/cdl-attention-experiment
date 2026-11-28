"""Frozen-weight last-component interventions on held, paired byte streams.

Every comparison uses the same input bytes and unrestricted answer logits.
Hooks alter the selected mechanism only for the counterfactual forward pass.
"""
import hashlib
from contextlib import ExitStack, contextmanager

import torch

from panels import DEV_PANELS, _batch_bytes, _english_world, _rng, _world


def replace_component(arm, side, stack):
    def hook(module, fn):
        handle = module.register_forward_hook(lambda mod, inp, out: fn(out))
        stack.callback(handle.remove)

    if arm == "D":
        # The extra delta bank starts empty on each independent episode.
        # Zero key, query and value keeps that bank empty for the entire stream.
        hook(side.key_query_value, torch.zeros_like)
    elif arm == "E":
        hook(side.event, lambda out: torch.full_like(out, 40))
    elif arm == "F":
        def cycle(out):
            times = torch.arange(out.size(1), device=out.device) % out.size(-1)
            logits = torch.full_like(out, -40)
            return logits.scatter_(-1, times[None, :, None].expand_as(out[..., :1]), 40)
        hook(side.router, cycle)
    elif arm == "G":
        hook(side.reliability, lambda out: torch.full_like(out, 40))
    elif arm == "H":
        def silence_geometry(out):
            # Route each selected geometric operation to identity/null. Leave
            # the route for every other token untouched at this layer's input.
            active = out.argmax(-1) == 3
            logits = out.clone()
            logits[..., 0] = torch.where(active, out[..., 3] + 1, out[..., 0])
            return logits
        hook(side.router, silence_geometry)


@contextmanager
def instrument(model, altered):
    """Restore all hooks, measurement flags and transient stats on exit."""
    saved = [(s.measure, s.stats, s.soft_cost) for s in model.parent.sides]
    with ExitStack() as stack:
        try:
            for side in model.parent.sides:
                side.measure = True
                if altered:
                    replace_component(model.arm, side, stack)
            yield
        finally:
            for side, (enabled, stats, cost) in zip(model.parent.sides, saved):
                side.measure, side.stats, side.soft_cost = enabled, stats, cost


def summary(logits, labels):
    correct = logits.argmax(-1).eq(labels)
    return {
        "both_correct": int(correct.view(-1, 2).all(-1).sum()),
        "answers_correct": int(correct.sum()),
        "answers": labels.numel(),
        "answer_nll_nats": torch.nn.functional.cross_entropy(logits, labels).item(),
    }


def score_confidence(scores, labels, selected):
    result = {}
    for name, mask in (("all", torch.ones_like(selected)), ("writes", selected)):
        p, y = scores[mask], labels[mask]
        result[name] = {
            "events": p.numel(), "reliable_events": int(y.sum()),
            "brier": (p - y).square().mean().item() if p.numel() else None,
            "constant_half_brier": 0.25 if p.numel() else None,
            "mean_score_reliable": p[y == 1].mean().item() if (y == 1).any() else None,
            "mean_score_unreliable": p[y == 0].mean().item() if (y == 0).any() else None,
        }
    return result


def confidence(model, samples):
    device = next(model.parameters()).device
    x = _batch_bytes(samples).to(device)
    # Exactly two claims concern the queried key, one ! and one ~.
    # Score the byte containing the claim value, after source and key are known.
    qkey = x[:, 16:17]
    relevant = x[:, 1:15:3].eq(qkey)
    labels = x[:, :15:3].eq(ord("!"))[relevant].float()
    if relevant.sum() != 2 * len(samples) or labels.sum() != len(samples):
        raise RuntimeError("Conflict panel format no longer matches the confidence audit")
    observations = []
    for _ in model.parent.sides:
        observations.append({})
    with ExitStack() as stack:
        for side, obs in zip(model.parent.sides, observations):
            for key, module in [("c", side.reliability), ("event", side.event), ("route", side.router)]:
                def capture(mod, args, out, obs=obs, key=key):
                    obs[key] = out[:, 2:15:3][relevant]
                handle = module.register_forward_hook(capture)
                stack.callback(handle.remove)
        model(x)
    rows = []
    for obs in observations:
        wrote = obs["route"].argmax(-1).eq(2) & obs["event"][:, 1].ge(0)
        rows.append(score_confidence(obs["c"].squeeze(-1).sigmoid(), labels, wrote))
    return rows


@torch.inference_mode()
def assess_interventions(model, worlds=24):
    """Compare frozen D–H components at fixed data. Uses integer counts per panel."""
    arm = getattr(model, "arm", "")
    if arm not in ("D", "E", "F", "G", "H"):
        return {}
    if worlds < 1:
        raise ValueError("worlds must be positive")
    training = model.training
    model.eval()
    device = next(model.parameters()).device
    result = {"arm": arm, "worlds_per_panel": worlds, "panels": {},
              "intervention": {
                  "D": "disable additional contextual delta bank",
                  "E": "open both state update gates",
                  "F": "cycle null/slow/memory at fixed equal frequency (selection and utilization change)",
                  "G": "set reliability factor to one",
                  "H": "redirect selected geometry to null",
              }[arm]}
    try:
        for index, panel in enumerate(DEV_PANELS):
            rng = _rng(610003, index, "dev")
            if panel.startswith("english_"):
                episodes = [_english_world(rng, panel, 2 if panel == "english_held_paraphrase" else i % 2)
                            for i in range(worlds)]
            else:
                episodes = [_world(rng, panel) for _ in range(worlds)]
            labels = torch.tensor([v for _, _, a, b in episodes for v in (a, b)])
            samples = [s for a, b, _, _ in episodes for s in (a, b)]
            digest = hashlib.sha256()
            for s, y in zip(samples, labels.tolist()):
                digest.update(len(s).to_bytes(4, "little") + s + bytes([y]))
            versions = []
            routes = []
            for altered in (False, True):
                logits = []
                counters = [{} for _ in model.parent.sides]
                with instrument(model, altered):
                    for start in range(0, len(samples), 16):
                        span = samples[start:start + 16]
                        # Group episodes by prompt length so route counts exclude padding.
                        ordered = [None] * len(span)
                        for length in sorted(set(map(len, span))):
                            idx = [i for i, s in enumerate(span) if len(s) == length]
                            ids = _batch_bytes([span[i] for i in idx]).to(device)
                            output, state = model(ids, carry=True)
                            last = output[:, -1].float().cpu()
                            for dest, side_state in zip(counters, state[2]):
                                for name, value in zip(("context", "slow", "delta_bank", "geometry"), side_state):
                                    if value is not None:
                                        key = f"{name}_terminal_sum_squares"
                                        dest[key] = dest.get(key, 0.) + value.double().square().sum().item()
                            if not torch.isfinite(last).all():
                                raise RuntimeError("Nonfinite intervention logits")
                            for i, row in zip(idx, last):
                                ordered[i] = row
                            for dest, stats in zip(counters, model.routing_metrics()):
                                for k, v in stats.items():
                                    dest[k] = dest.get(k, 0) + int(v)
                        logits.extend(ordered)
                versions.append(torch.stack(logits))
                routes.append(counters)
            original, altered = versions
            pre = original.argmax(-1).eq(labels).view(-1, 2).all(-1)
            post = altered.argmax(-1).eq(labels).view(-1, 2).all(-1)
            result["panels"][panel] = {
                "sha256": digest.hexdigest(), "intact": summary(original, labels),
                "altered": summary(altered, labels),
                "pair_gained": int((~pre & post).sum()), "pair_lost": int((pre & ~post).sum()),
                "max_logit_change": (original - altered).abs().max().item(),
                "terminal_states_and_selected_operations_by_layer": {"intact": routes[0], "altered": routes[1]},
            }
            if arm in ("G", "H") and panel == "conflict":
                with instrument(model, False):
                    result["digit_time_source_reliability_per_layer"] = confidence(model, samples[::2])
    finally:
        model.train(training)
    return result
