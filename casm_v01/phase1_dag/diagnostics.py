"""Mechanical Gates 0-6 for the Phase-1 CASM contract."""
from __future__ import annotations
import math
import torch

def _batch_x(episodes,device):
    n=max(len(e.inputs) for e in episodes)
    return torch.tensor([list(e.input_values)+[0]*(n-len(e.inputs)) for e in episodes],dtype=torch.float32,device=device)

def _execute_with_gates(model,ep,x,gates):
    vals=torch.zeros(model.max_nodes,device=x.device); vals[list(ep.inputs)]=x[:len(ep.inputs)]; by_dst={}
    for k,e in enumerate(ep.candidate_edges): by_dst.setdefault((e.dst,e.port),[]).append((k,e.src))
    for node in ep.nodes[:ep.active_count]:
        if node.op.value=="INPUT": continue
        args=[]
        for p in range(node.arity):
            items=by_dst[(node.index,p)]; den=sum(gates[k] for k,s in items)+1e-6; args.append(sum(gates[k]*vals[s] for k,s in items)/den)
        if node.op.value=="NOT": vals[node.index]=1-args[0]
        else:
            a,b=args
            if node.op.value=="AND": vals[node.index]=a*b
            elif node.op.value=="OR": vals[node.index]=a+b-a*b
            elif node.op.value=="XOR": vals[node.index]=a+b-2*a*b
    return vals[ep.output]

def gate0_viability(model,episodes):
    device=next(model.parameters()).device; x=_batch_x(episodes,device); _,g,_=model(episodes,x); nz=g[g!=0]; trace=model.last_node_values
    mean_g=float(nz.mean()); std_g=float(nz.std()); variances=[]
    for d in range(max(n.depth for e in episodes for n in e.nodes)+1):
        vals=[trace[b,node.index] for b,e in enumerate(episodes) for node in e.nodes[:e.active_count] if node.depth==d]
        if vals: variances.append(float(torch.stack(vals).var(unbiased=False)))
    finite=bool(torch.isfinite(trace).all() and torch.isfinite(g).all()); bounded=bool(trace.abs().max()<8)
    return {"pass":finite and bounded and .40<=mean_g<=.60 and std_g<.25,"mean_gate":mean_g,"std_gate":std_g,"depth_variance":variances,"max_abs_node_value":float(trace.abs().max())}

def gate1_causality(model,episode):
    device=next(model.parameters()).device; x=torch.tensor([episode.input_values],dtype=torch.float32,device=device); y,g,_=model([episode],x); true_idx=[k for k,e in enumerate(episode.candidate_edges) if (e.src,e.dst,e.port) in episode.true_edge_set]; k=max(true_idx,key=lambda i:float(g[0,i]))
    edited=g[0].detach().clone(); edited[k]=0; cf=_execute_with_gates(model,episode,x[0],edited); delta=abs(float(y.item())-float(cf.item()))
    return {"pass":delta>1e-7,"edge_index":k,"forward_delta":delta}

def gate2_router_gradient(model,episodes):
    model.zero_grad(set_to_none=True); device=next(model.parameters()).device; y,_,_=model(episodes,_batch_x(episodes,device)); target=torch.tensor([e.target for e in episodes],dtype=torch.float32,device=device); loss=torch.nn.functional.mse_loss(y,target); loss.backward(); norms=[p.grad.detach().norm().item() for n,p in model.named_parameters() if any(k in n for k in ("q.","k.","bias_relation")) and p.grad is not None]; norm=math.sqrt(sum(v*v for v in norms)) if norms else 0
    return {"pass":norm>0 and math.isfinite(norm),"router_grad_norm":norm,"loss":float(loss)}

def gate3_structural_sensitivity(model,ep_a,ep_b):
    device=next(model.parameters()).device; x=_batch_x([ep_a,ep_b],device); _,g,_=model([ep_a,ep_b],x); n=min(len(ep_a.candidate_edges),len(ep_b.candidate_edges)); diff=float((g[0,:n]-g[1,:n]).abs().mean())
    return {"pass":diff>1e-4,"mean_gate_difference":diff}

def gate4_counterfactual(model,episode):
    device=next(model.parameters()).device; x=torch.tensor([episode.input_values],dtype=torch.float32,device=device); target=torch.tensor([float(episode.target)],device=device); y,g,_=model([episode],x); base=float(torch.nn.functional.mse_loss(y,target)); deltas=[]
    for k,e in enumerate(episode.candidate_edges):
        if (e.src,e.dst,e.port) not in episode.true_edge_set: continue
        edited=g[0].detach().clone(); edited[k]=0; yy=_execute_with_gates(model,episode,x[0],edited); deltas.append((float(torch.nn.functional.mse_loss(yy.view(1),target))-base,k))
    best=max(deltas,default=(0,-1)); return {"pass":best[0]>1e-5,"max_true_edge_loss_delta":best[0],"edge_index":best[1]}

def gate5_compositional_ood(model,episode,source_op,target_op):
    device=next(model.parameters()).device; x=_batch_x([episode],device); _,g1,_=model([episode],x); _,g2,_=model([episode],x); vals=[]
    for k,e in enumerate(episode.candidate_edges):
        if episode.nodes[e.src].op.value==source_op and episode.nodes[e.dst].op.value==target_op: vals.append((float(g1[0,k]),(e.src,e.dst,e.port) in episode.true_edge_set))
    pos=[v for v,t in vals if t]; neg=[v for v,t in vals if not t]; sep=(sum(pos)/len(pos)-sum(neg)/len(neg)) if pos and neg else 0
    reproducible=bool(torch.allclose(g1,g2,atol=1e-7,rtol=0))
    return {"pass":bool(pos and neg and sep>0 and reproducible),"role_pair":f"{source_op}->{target_op}","positive_mean":sum(pos)/len(pos) if pos else None,"negative_mean":sum(neg)/len(neg) if neg else None,"separation":sep,"reproducible":reproducible}

def gate6_integrity(model,episodes):
    ok=tuple(model.alpha_eta.shape)==(2,) and not hasattr(model,"alpha_ij") and not hasattr(model,"edge_alpha") and not hasattr(model,"edge_alpha_table")
    alpha=model.c*torch.nn.functional.softplus(model.alpha_eta)
    ok=ok and bool(torch.isfinite(alpha).all()) and bool((alpha>=0).all())
    for ep in episodes:
        ok=ok and len(ep.parent_slots)==len(ep.nodes) and all(len(p)<=2 for p in ep.parent_slots)
        ok=ok and all(e.src<e.dst for e in ep.candidate_edges) and all(e.src<ep.active_count and e.dst<ep.active_count for e in ep.true_edges)
    return {"pass":bool(ok),"alpha_shape":tuple(model.alpha_eta.shape),"alpha_value":alpha.detach().cpu().tolist(),"edge_parameter_tables":False,"topological_order":bool(ok),"public_parent_slots":bool(ok)}
