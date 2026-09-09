from __future__ import annotations

import argparse, json, math, platform, random, subprocess
from pathlib import Path
import torch
from .run_stable_cardinality_executor import set_optimizer_lr
from .run_state_instantiation_credit import _selection_metrics, _capability_metrics, cosine_lr
from .state_instantiation_data import NUM_CANDIDATES, TRAIN_LIVE_CARDINALITIES, UNSEEN_LIVE_CARDINALITIES, make_state_instantiation_batch, training_live_cardinality_for_step
from .state_instantiation_reuse_merge import X20V_MODES, X20V_LEARNED_MODES, reuse_merge_loss_components, cloned_x20v_models, merge_gates, HARD_GATE_THRESHOLD

HARNESS_VERSION = "casm-x20v-reuse-merge-v0-2026-09-05"
STEPS=12000; BATCH_SIZE=128; TRAIN_DEPTH=12; EVAL_N=256; EVAL_BATCH_SIZE=64
LR_MAX=2e-3; LR_MIN=2e-4; WEIGHT_DECAY=1e-4; GRAD_CLIP_NORM=1.0


def _git_sha(): return subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip()

def train_step(model, opt, batch, mode, lr):
    set_optimizer_lr(opt, lr); opt.zero_grad(set_to_none=True)
    parts=reuse_merge_loss_components(model,batch,mode=mode)
    for k,v in parts.items():
        x=float(v.detach())
        if not math.isfinite(x) or x<0: raise RuntimeError(f"invalid {k}: {x}")
    parts["total_loss"].backward()
    gn=torch.nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP_NORM)
    if not math.isfinite(float(gn)): raise RuntimeError("non-finite gradient norm")
    opt.step(); return {**{k:float(v.detach()) for k,v in parts.items()},"grad_norm":float(gn),"lr":float(lr)}

@torch.no_grad()
def evaluate_mode(model, *, live_cardinality, depth, split, n, batch_size, seed):
    rows=[]; remaining=n; bi=0
    while remaining:
        size=min(batch_size,remaining)
        b=make_state_instantiation_batch(size,depth,seed+bi*1009,live_cardinality=live_cardinality,split=split)
        gates=model.soft_gates(b)
        hard_pred=model.executor.rollout_hard(b.program,gates)
        soft=model.executor.rollout_soft(b.program,gates)
        soft_pred=soft[:,:,:,:16].argmax(dim=-1)
        sel=_selection_metrics(gates,b.live_mask)
        _, reuse=merge_gates(super(type(model),model).soft_gates(b) if False else gates,b,structure_blind=model.x20v_mode=="reuse_merge_structure_blind")
        rows.append((_capability_metrics(hard_pred,b.program.target_states,b.live_mask),_capability_metrics(soft_pred,b.program.target_states,b.live_mask),sel,reuse,size))
        remaining-=size; bi+=1
    total=sum(r[4] for r in rows)
    def merge(i):
        return {k:sum(r[i][k]*r[4] for r in rows)/total for k in rows[0][i]}
    return {"hard":merge(0),"soft":merge(1),"selection":merge(2),"reuse":merge(3)}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--steps',type=int,default=STEPS); p.add_argument('--batch-size',type=int,default=BATCH_SIZE); p.add_argument('--train-depth',type=int,default=TRAIN_DEPTH); p.add_argument('--seed',type=int,required=True); p.add_argument('--eval-seed',type=int,required=True); p.add_argument('--eval-n',type=int,default=EVAL_N); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    if (a.steps,a.batch_size,a.train_depth,a.eval_n)!=(STEPS,BATCH_SIZE,TRAIN_DEPTH,EVAL_N): raise ValueError('CASM-X20V frozen run contract mismatch')
    torch.manual_seed(a.seed); random.seed(a.seed); torch.set_num_threads(max(1,min(4,torch.get_num_threads())))
    models=cloned_x20v_models(); opts={m:torch.optim.AdamW(models[m].parameters(),lr=LR_MAX,weight_decay=WEIGHT_DECAY) for m in X20V_MODES}
    counts={m:models[m].parameter_count() for m in X20V_MODES}; trainable={m:models[m].trainable_parameter_count() for m in X20V_MODES}
    assert len(set(counts.values()))==1 and len(set(trainable.values()))==1
    cardinality_counts={str(n):0 for n in TRAIN_LIVE_CARDINALITIES}; minimum_task={m:float('inf') for m in X20V_MODES}; final_task={m:float('nan') for m in X20V_MODES}; history=[]; max_gn={m:0.0 for m in X20V_MODES}
    for step in range(1,STEPS+1):
        nlive=training_live_cardinality_for_step(step); cardinality_counts[str(nlive)]+=1
        b=make_state_instantiation_batch(BATCH_SIZE,TRAIN_DEPTH,a.seed*1000003+step*97,live_cardinality=nlive,split='train'); lr=cosine_lr(step); row={'step':step,'live_cardinality':nlive}
        for m in X20V_MODES:
            row[m]=train_step(models[m],opts[m],b,m,lr); minimum_task[m]=min(minimum_task[m],row[m]['task_loss']); final_task[m]=row[m]['task_loss']; max_gn[m]=max(max_gn[m],row[m]['grad_norm'])
        if step==1 or step%1000==0 or step==STEPS: history.append(row); print(json.dumps(row),flush=True)
    for m in models.values(): m.eval()
    suites=[('iid_depth_12','iid',12),('composition_depth_24','composition',24),('stress_depth_48','composition',48),('stress_depth_96','composition',96)]
    evaluation={}
    for nlive in (*TRAIN_LIVE_CARDINALITIES,*UNSEEN_LIVE_CARDINALITIES):
        evaluation[str(nlive)]={}
        for si,(suite,split,depth) in enumerate(suites):
            evaluation[str(nlive)][suite]={"split":split,"depth":depth}
            for m in X20V_MODES:
                evaluation[str(nlive)][suite][m]=evaluate_mode(models[m],live_cardinality=nlive,depth=depth,split=split,n=EVAL_N,batch_size=EVAL_BATCH_SIZE,seed=a.eval_seed+nlive*1000003+si*100003)
    report={'harness_version':HARNESS_VERSION,'git_sha':_git_sha(),'python_version':platform.python_version(),'torch_version':torch.__version__,'seed':a.seed,'eval_seed':a.eval_seed,'steps':STEPS,'batch_size':BATCH_SIZE,'train_depth':TRAIN_DEPTH,'eval_n':EVAL_N,'candidate_count':NUM_CANDIDATES,'training_live_cardinalities':list(TRAIN_LIVE_CARDINALITIES),'unseen_live_cardinalities':list(UNSEEN_LIVE_CARDINALITIES),'training_cardinality_counts':cardinality_counts,'parameters':counts,'trainable_parameters':trainable,'minimum_task_loss':minimum_task,'final_task_loss':final_task,'maximum_post4000_grad_norm':max_gn,'objective_contract':{'no_reuse_control':'0.5*(0.5*A_hard+0.5*A_soft)+0.5*L_local+0.05*S','reuse_merge_graph':'0.5*(0.5*A_hard+0.5*A_soft)+0.5*L_local+0.05*S','reuse_merge_structure_blind':'0.5*(0.5*A_hard+0.5*A_soft)+0.5*L_local+0.05*S','same_final_answer_target':True,'counterfactual_path_adds_no_labels':True},'constructor_contract':{'learned_live_mask_input':False,'learned_active_cardinality_input':False,'per_candidate_parameter_table':False,'hard_gate_threshold':HARD_GATE_THRESHOLD,'straight_through_forward_binary':True,'reuse_graph_uses_supplied_candidate_program_tensors':True,'structure_blind_removes_dependency_version_compatibility':True},'supervision_contract':{'final_answer_only':True,'live_mask_loss':False,'cardinality_loss':False,'merge_label_loss':False,'hidden_state_loss':False,'intermediate_target_loss':False,'top_k':False,'lookup_table':False,'post_hoc_threshold':False},'reuse_contract':{'duplicate_active_identity_rate_max':0.05,'merge_precision_min':0.95,'merge_recall_min':0.90,'mean_hard_record_count_error_max':0.25,'hard_gate_threshold':0.5},'training_history':history,'evaluation':evaluation}
    for v in minimum_task.values():
        if not math.isfinite(v): raise RuntimeError('non-finite report')
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__': main()
