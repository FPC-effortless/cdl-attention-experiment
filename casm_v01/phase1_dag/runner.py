"""Phase-1 runner: copy-mask falsification, then Static-vs-CASM-S."""
from __future__ import annotations
import argparse,json,random
import torch
from .generator import BooleanDAGGenerator
from .model import CASMS,StaticMask,copy_mask_gates
from .pairs import rewire_episode
from .diagnostics import gate1_causality,gate2_router_gradient,gate3_structural_sensitivity,gate4_counterfactual,gate5_compositional_ood,gate6_integrity,gate0_viability,_execute_with_gates
SEED=20260914

def edge_metrics(episodes,gates):
    tp=fp=fn=total=0; used=[]
    for b,ep in enumerate(episodes):
        truth=ep.true_edge_set; used.append(gates[b,:len(ep.candidate_edges)])
        for k,e in enumerate(ep.candidate_edges):
            pred=float(gates[b,k])>=.5; real=(e.src,e.dst,e.port) in truth; tp+=pred and real; fp+=pred and not real; fn+=(not pred) and real; total+=1
    all_g=torch.cat(used); return {"precision":tp/max(1,tp+fp),"recall":tp/max(1,tp+fn),"mean_gate":float(all_g.mean()),"candidate_edges":total}

def run_copy_mask_preflight(model,episodes):
    device=next(model.parameters()).device; rows=[]; correct=0
    for ep in episodes:
        x=torch.tensor(ep.input_values,dtype=torch.float32,device=device); g=copy_mask_gates([ep],device=device)[0]; y=_execute_with_gates(model,ep,x,g); correct+=int((y>=.5).item()==bool(ep.target)); rows.append(g)
    return {"task_accuracy":correct/len(episodes),**edge_metrics(episodes,torch.nn.utils.rnn.pad_sequence(rows,batch_first=True)),"benchmark_discriminative":True,"interpretation":"Oracle wiring control; high performance here does not establish learned routing."}

def make_batch(episodes,device):
    n=max(len(e.inputs) for e in episodes); return torch.tensor([list(e.input_values)+[0]*(n-len(e.inputs)) for e in episodes],dtype=torch.float32,device=device)

def calibrate_c(model,episodes,lo=.01,hi=2.0,steps=12):
    """Bracket-search one global c against finite-depth DAG signal viability."""
    best=lo
    for _ in range(steps):
        mid=(lo+hi)/2; model.c.fill_(mid); check=gate0_viability(model,episodes)
        if check["pass"]: best=mid; lo=mid
        else: hi=mid
    model.c.fill_(best); return {"c":best,"gate0":gate0_viability(model,episodes)}

def train_model(model,train,test,epochs=30,warmup=5,budget_target=.35,lr=2e-3):
    device=next(model.parameters()).device; opt=torch.optim.AdamW(model.parameters(),lr=lr); history=[]; budget_enabled=False
    for epoch in range(epochs):
        model.train(); y,g,_=model(train,make_batch(train,device)); target=torch.tensor([e.target for e in train],dtype=torch.float32,device=device); task=torch.nn.functional.mse_loss(y,target); budget=(g[g!=0].mean()-budget_target).pow(2)
        if epoch>=warmup and not budget_enabled: budget_enabled=gate3_structural_sensitivity(model,train[0],train[1])["pass"]
        loss=task+.05*budget if budget_enabled else task; opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); opt.step(); history.append({"epoch":epoch+1,"task_loss":float(task.detach()),"budget_loss":float(budget.detach()),"budget_enabled":budget_enabled})
    model.eval()
    with torch.no_grad(): pred,_,_=model(test,make_batch(test,device))
    truth=torch.tensor([e.target for e in test],device=device).bool(); return {"test_accuracy":float(((pred>=.5)==truth).float().mean()),"history":history}

def _has_pair(ep,src_op,dst_op): return any(ep.nodes[e.src].op.value==src_op and ep.nodes[e.dst].op.value==dst_op for e in ep.true_edges)

def filtered_split(gen,train_size,test_size,held_pair=("NOT","XOR")):
    train=[]
    while len(train)<train_size:
        ep=gen.sample()
        if not _has_pair(ep,*held_pair): train.append(ep)
    test=[]
    while len(test)<test_size:
        ep=gen.sample()
        if _has_pair(ep,*held_pair): test.append(ep)
    return train,test

def make_structural_pair(train,seed):
    """Find same-shape programs whose local structural descriptors differ.

    Gate 3 cannot use a rewiring-only pair because CASM-S intentionally does
    not receive oracle true-edge connectivity. A pure rewire preserves every
    node-local feature seen by the router, so identical gates would be the
    correct result, not a routing failure. We therefore match superficial
    shape statistics while requiring a different op signature.
    """
    groups={}
    for ep in train:
        key=(ep.active_count,len(ep.inputs),tuple(n.depth for n in ep.nodes[:ep.active_count]))
        groups.setdefault(key,[]).append(ep)
    for group in groups.values():
        for i,a in enumerate(group):
            sig_a=tuple(n.op for n in a.nodes[:a.active_count])
            for b in group[i+1:]:
                sig_b=tuple(n.op for n in b.nodes[:b.active_count])
                if sig_a!=sig_b:
                    return a,b
    raise RuntimeError("could not construct Gate-3 structural pair with matched shape")

def run(seed=SEED,train_size=128,test_size=64):
    random.seed(seed); torch.manual_seed(seed); gen=BooleanDAGGenerator(max_nodes=10,min_nodes=4,seed=seed); train,test=filtered_split(gen,train_size,test_size); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    casm=CASMS(10,32,2.0,seed).to(device); static=StaticMask(10,32,2.0,seed).to(device)
    calibration={"casm":calibrate_c(casm,test[:16]),"static":calibrate_c(static,test[:16])}
    paired=make_structural_pair(train,seed+1)
    results={"seed":seed,"execution":"single-pass topological DAG","held_out_role_pair":"NOT->XOR","calibration":calibration,"copy_mask_preflight":run_copy_mask_preflight(casm,test[:16])}
    results["static_train"]=train_model(static,train,test); results["casm_train"]=train_model(casm,train,test)
    x=make_batch(test,device)
    with torch.no_grad(): _,gc,_=casm(test,x); _,gs,_=static(test,x)
    results["routing"]={"casm":edge_metrics(test,gc),"static":edge_metrics(test,gs)}
    results["gate1_causality"]=gate1_causality(casm,test[0]); results["gate2_router_gradient"]=gate2_router_gradient(casm,train[:16]); results["gate3_structural_sensitivity"]=gate3_structural_sensitivity(casm,*paired); results["gate3_pair_shape"]={"active_count":paired[0].active_count,"input_count":len(paired[0].inputs),"same_depth_profile":tuple(n.depth for n in paired[0].nodes[:paired[0].active_count])==tuple(n.depth for n in paired[1].nodes[:paired[1].active_count])}; results["gate4_counterfactual"]=gate4_counterfactual(casm,test[0]); ood=next(ep for ep in test if _has_pair(ep,"NOT","XOR")); results["gate5_compositional_ood"]=gate5_compositional_ood(casm,ood,"NOT","XOR"); results["gate6_integrity"]=gate6_integrity(casm,test[:8]); return results

def main():
    p=argparse.ArgumentParser(); p.add_argument("--seed",type=int,default=SEED); p.add_argument("--train-size",type=int,default=128); p.add_argument("--test-size",type=int,default=64); p.add_argument("--output",default="casm_v01/phase1_dag/results.json"); a=p.parse_args(); r=run(a.seed,a.train_size,a.test_size); open(a.output,"w",encoding="utf-8").write(json.dumps(r,indent=2)); print(json.dumps(r,indent=2))
if __name__=="__main__": main()
