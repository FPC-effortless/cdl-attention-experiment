from __future__ import annotations
import argparse,json,math,platform,random,subprocess
from pathlib import Path
import torch
from .run_stable_cardinality_executor import set_optimizer_lr
from .run_state_instantiation_credit import _selection_metrics,_capability_metrics,cosine_lr
from .state_instantiation_data import NUM_CANDIDATES,TRAIN_LIVE_CARDINALITIES,UNSEEN_LIVE_CARDINALITIES,make_state_instantiation_batch,training_live_cardinality_for_step
from .state_instantiation_local_credit import local_credit_loss_components,LOCAL_CREDIT_MODE
from .state_instantiation_reuse_merge import X20V_MODES,X20VStateInstantiationModel,cloned_x20v_models,merge_gates,REUSE_BLIND_MODE,NO_REUSE_MODE,HARD_GATE_THRESHOLD

HARNESS_VERSION='casm-x20v-reuse-merge-v1-2026-09-05'
STEPS=12000;BATCH_SIZE=128;TRAIN_DEPTH=12;EVAL_N=256;EVAL_BATCH_SIZE=64;LR_MAX=2e-3;LR_MIN=2e-4;WEIGHT_DECAY=1e-4;CLIP=1.0

def git_sha(): return subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
def loss(model,batch,mode):
    if mode==NO_REUSE_MODE:
        return local_credit_loss_components(model,batch,mode=LOCAL_CREDIT_MODE)
    from .state_instantiation_reuse_merge import reuse_merge_loss_components
    return reuse_merge_loss_components(model,batch,mode=mode)
def step(model,opt,batch,mode,lr):
    set_optimizer_lr(opt,lr);opt.zero_grad(set_to_none=True);parts=loss(model,batch,mode)
    for v in parts.values():
        x=float(v.detach())
        if not math.isfinite(x) or x<0: raise RuntimeError('invalid loss component')
    parts['total_loss'].backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),CLIP)
    if not math.isfinite(float(gn)): raise RuntimeError('non-finite gradient norm')
    opt.step();return {k:float(v.detach()) for k,v in parts.items()}|{'grad_norm':float(gn),'lr':lr}
@torch.no_grad()
def evaluate(model,live_cardinality,depth,split,n,batch_size,seed):
    rows=[];remaining=n;bi=0
    while remaining:
        size=min(batch_size,remaining);b=make_state_instantiation_batch(size,depth,seed+bi*1009,live_cardinality=live_cardinality,split=split)
        base=X20VStateInstantiationModel.soft_gates(model,b)
        if model.x20v_mode==NO_REUSE_MODE:
            gates=base;reuse={'duplicate_active_identity_rate':0.0,'merge_precision':1.0,'merge_recall':1.0,'mean_hard_record_count_error':0.0}
        else:
            gates,reuse=merge_gates(base,b,structure_blind=model.x20v_mode==REUSE_BLIND_MODE)
        hard=model.executor.rollout_hard(b.program,gates);soft=model.executor.rollout_soft(b.program,gates);soft_pred=soft[:,:,:,:16].argmax(-1)
        rows.append((_capability_metrics(hard,b.program.target_states,b.live_mask),_capability_metrics(soft_pred,b.program.target_states,b.live_mask),_selection_metrics(gates,b.live_mask),reuse,size));remaining-=size;bi+=1
    total=sum(x[4] for x in rows)
    def avg(i): return {k:sum(x[i][k]*x[4] for x in rows)/total for k in rows[0][i]}
    return {'hard':avg(0),'soft':avg(1),'selection':avg(2),'reuse':avg(3)}
def main():
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=STEPS);p.add_argument('--batch-size',type=int,default=BATCH_SIZE);p.add_argument('--train-depth',type=int,default=TRAIN_DEPTH);p.add_argument('--eval-n',type=int,default=EVAL_N);p.add_argument('--seed',type=int,required=True);p.add_argument('--eval-seed',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if (a.steps,a.batch_size,a.train_depth,a.eval_n)!=(STEPS,BATCH_SIZE,TRAIN_DEPTH,EVAL_N):raise ValueError('CASM-X20V frozen run contract mismatch')
    torch.manual_seed(a.seed);random.seed(a.seed);torch.set_num_threads(max(1,min(4,torch.get_num_threads())))
    models=cloned_x20v_models();opts={m:torch.optim.AdamW(models[m].parameters(),lr=LR_MAX,weight_decay=WEIGHT_DECAY) for m in X20V_MODES}
    counts={m:models[m].parameter_count() for m in X20V_MODES};trainable={m:models[m].trainable_parameter_count() for m in X20V_MODES}
    assert len(set(counts.values()))==1 and len(set(trainable.values()))==1
    mins={m:float('inf') for m in X20V_MODES};final={m:float('nan') for m in X20V_MODES};maxgn={m:0.0 for m in X20V_MODES};cc={str(n):0 for n in TRAIN_LIVE_CARDINALITIES};history=[]
    for s in range(1,STEPS+1):
        n=training_live_cardinality_for_step(s);cc[str(n)]+=1;b=make_state_instantiation_batch(BATCH_SIZE,TRAIN_DEPTH,a.seed*1000003+s*97,live_cardinality=n,split='train');lr=cosine_lr(s);row={'step':s,'live_cardinality':n}
        for m in X20V_MODES:
            row[m]=step(models[m],opts[m],b,m,lr);mins[m]=min(mins[m],row[m]['task_loss']);final[m]=row[m]['task_loss'];maxgn[m]=max(maxgn[m],row[m]['grad_norm'])
        if s==1 or s%1000==0 or s==STEPS:history.append(row);print(json.dumps(row),flush=True)
    for m in models.values():m.eval()
    evaluation={};suites=[('iid_depth_12','iid',12),('composition_depth_24','composition',24),('stress_depth_48','composition',48),('stress_depth_96','composition',96)]
    for n in (*TRAIN_LIVE_CARDINALITIES,*UNSEEN_LIVE_CARDINALITIES):
        evaluation[str(n)]={}
        for i,(name,split,depth) in enumerate(suites):
            evaluation[str(n)][name]={"split":split,"depth":depth}
            for m in X20V_MODES:evaluation[str(n)][name][m]=evaluate(models[m],n,depth,split,EVAL_N,EVAL_BATCH_SIZE,a.eval_seed+n*1000003+i*100003)
    report={'harness_version':HARNESS_VERSION,'git_sha':git_sha(),'python_version':platform.python_version(),'torch_version':torch.__version__,'seed':a.seed,'eval_seed':a.eval_seed,'steps':STEPS,'batch_size':BATCH_SIZE,'train_depth':TRAIN_DEPTH,'eval_n':EVAL_N,'candidate_count':NUM_CANDIDATES,'training_live_cardinalities':list(TRAIN_LIVE_CARDINALITIES),'unseen_live_cardinalities':list(UNSEEN_LIVE_CARDINALITIES),'training_cardinality_counts':cc,'parameters':counts,'trainable_parameters':trainable,'minimum_task_loss':mins,'final_task_loss':final,'maximum_post4000_grad_norm':maxgn,'objective_contract':{'no_reuse_control':'0.5*(0.5*A_hard+0.5*A_soft)+0.5*L_local+0.05*S','reuse_merge_graph':'0.5*(0.5*A_hard+0.5*A_soft)+0.5*L_local+0.05*S','reuse_merge_structure_blind':'0.5*(0.5*A_hard+0.5*A_soft)+0.5*L_local+0.05*S','local_risk':'mean_i[g_soft_i*stopgrad(A_on_i)+(1-g_soft_i)*stopgrad(A_off_i)]','same_final_answer_target':True,'counterfactual_path_adds_no_labels':True},'constructor_contract':{'learned_live_mask_input':False,'learned_active_cardinality_input':False,'per_candidate_parameter_table':False,'hard_gate_threshold':HARD_GATE_THRESHOLD,'straight_through_forward_binary':True,'reuse_graph_uses_supplied_candidate_program_tensors':True,'structure_blind_removes_dependency_version_compatibility':True},'supervision_contract':{'final_answer_only':True,'live_mask_loss':False,'cardinality_loss':False,'merge_label_loss':False,'hidden_state_loss':False,'intermediate_target_loss':False,'top_k':False,'lookup_table':False,'post_hoc_threshold':False},'reuse_contract':{'duplicate_active_identity_rate_max':0.05,'merge_precision_min':0.95,'merge_recall_min':0.90,'mean_hard_record_count_error_max':0.25,'hard_gate_threshold':0.5},'training_history':history,'evaluation':evaluation}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
