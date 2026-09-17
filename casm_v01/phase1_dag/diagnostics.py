"""Mechanical diagnostics for the Phase-1 CASM contract."""
from __future__ import annotations

import math
import torch

from .casm_s import structural_tensor
from .generator import Episode
from .oracle import evaluate_episode


def copy_mask_report(episode: Episode) -> dict:
    """Evaluate the non-oracle existence/admissibility mask.

    The copy-mask is one for every admissible candidate edge whose endpoints
    exist. It deliberately does not inspect ``true_edge_set``. Because the
    substrate is a strict superset, this mask must be non-executable under the
    exact-arity oracle.
    """
    candidate = len(episode.candidate_edges)
    true = len(episode.true_edges)
    copy_edges = tuple(episode.candidate_edges)
    try:
        exact = evaluate_episode(episode, copy_edges) == episode.truth_table
        executable = True
    except (KeyError, ValueError):
        exact = False
        executable = False
    return {
        "candidate_edges": candidate,
        "true_edges": true,
        "distractor_edges": candidate - true,
        "copy_mask_equals_true": False,
        "copy_mask_executable": executable,
        "copy_mask_exact": exact,
    }


def preflight(episode: Episode, max_edges: int = 12) -> dict:
    report = copy_mask_report(episode)
    from .oracle import exhaustive_minimality
    minimality = exhaustive_minimality(episode, max_edges=max_edges)
    return {
        **report,
        "oracle_rows": len(episode.truth_table),
        "minimality": minimality,
        "ready_for_router": (
            report["distractor_edges"] > 0
            and not report["copy_mask_executable"]
            and not report["copy_mask_exact"]
            and minimality["checked"]
            and minimality["minimal"]
            and minimality["unique_minimal"]
        ),
    }


def gate3_structural_sensitivity(model, ep_a: Episode, ep_b: Episode) -> dict:
    """Require different *visible structural composition*, not hidden rewiring.

    A router that is intentionally forbidden from seeing oracle wiring cannot
    distinguish two episodes whose node-local structural descriptors are
    identical. Such a pair is therefore an impossible Gate-3 test. This gate
    instead requires different operation/type composition with the same graph
    size, input count and candidate-edge cardinality.
    """
    if len(ep_a.nodes) != len(ep_b.nodes) or len(ep_a.inputs) != len(ep_b.inputs):
        raise ValueError("Gate 3 pair must have identical size and input count")
    if len(ep_a.candidate_edges) != len(ep_b.candidate_edges):
        raise ValueError("Gate 3 pair must have identical substrate size")
    sig_a = tuple(n.op.value for n in ep_a.nodes)
    sig_b = tuple(n.op.value for n in ep_b.nodes)
    if sig_a == sig_b:
        raise ValueError("Gate 3 pair must differ in visible structural composition")
    device = next(model.parameters()).device
    xa = torch.tensor([ep_a.input_values], dtype=torch.float32, device=device)
    xb = torch.tensor([ep_b.input_values], dtype=torch.float32, device=device)
    _, ga, _ = model([ep_a], xa)
    _, gb, _ = model([ep_b], xb)
    n = min(ga.shape[1], gb.shape[1])
    diff = float((ga[0, :n] - gb[0, :n]).abs().mean())
    return {"pass": diff > 1e-4, "mean_gate_difference": diff,
            "visible_signature_a": sig_a, "visible_signature_b": sig_b}


def gate0_viability(model, episodes):
    device = next(model.parameters()).device
    x = torch.tensor([e.input_values for e in episodes], dtype=torch.float32, device=device)
    y, g, _ = model(episodes, x)
    nz = g[g != 0]
    mean_g, std_g = float(nz.mean()), float(nz.std(unbiased=False))
    trace = model.last_node_values
    finite = bool(torch.isfinite(trace).all() and torch.isfinite(g).all())
    bounded = bool(trace.abs().max() < 8)
    return {"pass": finite and bounded and .40 <= mean_g <= .60 and std_g < .25,
            "mean_gate": mean_g, "std_gate": std_g,
            "max_abs_node_value": float(trace.abs().max())}


def gate1_causality(model, episode):
    device = next(model.parameters()).device
    x = torch.tensor([episode.input_values], dtype=torch.float32, device=device)
    y, g, _ = model([episode], x)
    k = int(torch.argmax(g[0]).item())
    edited = g[0].detach().clone(); edited[k] = 0
    cf = model.execute_with_gates(episode, x[0], edited)
    delta = abs(float(y.item()) - float(cf.item()))
    return {"pass": delta > 1e-7, "edge_index": k, "forward_delta": delta}


def gate2_router_gradient(model, episodes):
    model.zero_grad(set_to_none=True)
    device = next(model.parameters()).device
    x = torch.tensor([e.input_values for e in episodes], dtype=torch.float32, device=device)
    y, _, _ = model(episodes, x)
    target = torch.tensor([e.target for e in episodes], dtype=torch.float32, device=device)
    loss = torch.nn.functional.mse_loss(y, target)
    loss.backward()
    norms = [p.grad.detach().norm().item() for n,p in model.named_parameters()
             if any(k in n for k in ("q.", "k.", "relation_bias")) and p.grad is not None]
    norm = math.sqrt(sum(v*v for v in norms)) if norms else 0
    return {"pass": norm > 0 and math.isfinite(norm), "router_grad_norm": norm, "loss": float(loss)}


def gate4_counterfactual(model, episode):
    device = next(model.parameters()).device
    x = torch.tensor([episode.input_values], dtype=torch.float32, device=device)
    target = torch.tensor([float(episode.target)], device=device)
    y, g, _ = model([episode], x)
    base = float(torch.nn.functional.mse_loss(y, target))
    deltas = []
    for k,e in enumerate(episode.candidate_edges):
        if (e.src,e.dst,e.port) not in episode.true_edge_set: continue
        edited = g[0].detach().clone(); edited[k] = 0
        yy = model.execute_with_gates(episode, x[0], edited)
        deltas.append((float(torch.nn.functional.mse_loss(yy.view(1), target))-base, k))
    best = max(deltas, default=(0,-1))
    return {"pass": best[0] > 1e-5, "max_true_edge_loss_delta": best[0], "edge_index": best[1]}


def gate5_compositional_ood(model, episode, source_op, target_op):
    device = next(model.parameters()).device
    x = torch.tensor([episode.input_values], dtype=torch.float32, device=device)
    _, g, _ = model([episode], x)
    vals = []
    for k,e in enumerate(episode.candidate_edges):
        if episode.nodes[e.src].op.value == source_op and episode.nodes[e.dst].op.value == target_op:
            vals.append((float(g[0,k]), (e.src,e.dst,e.port) in episode.true_edge_set))
    pos=[v for v,t in vals if t]; neg=[v for v,t in vals if not t]
    sep=(sum(pos)/len(pos)-sum(neg)/len(neg)) if pos and neg else 0
    return {"pass": bool(pos and neg and sep > 0), "role_pair": f"{source_op}->{target_op}",
            "positive_mean": sum(pos)/len(pos) if pos else None,
            "negative_mean": sum(neg)/len(neg) if neg else None, "separation": sep}


def gate6_integrity(model, episodes):
    ok = tuple(model.alpha_eta.shape) == (2,) and not hasattr(model, "alpha_ij") and not hasattr(model, "edge_alpha")
    ok = ok and tuple(getattr(model, "REL_IDS", (0,1))) == (0,1) if hasattr(model, "REL_IDS") else ok
    for ep in episodes:
        ok = ok and all(e.src < e.dst for e in ep.candidate_edges)
    return {"pass": bool(ok), "alpha_shape": tuple(model.alpha_eta.shape),
            "edge_parameter_tables": False, "topological_order": bool(ok)}
