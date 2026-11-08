"""Counterfactual-credit diagnostics, explicitly distinguished by their scope."""
import random
import torch
from ..versioned_baselines.study import take


@torch.inference_mode()
def credit_diagnostics(model, data, seed, examples=32, repeats=4):
    # Import lazily to avoid module initialization cycles.
    from .run import intervention, per_loss
    batch=take(data,slice(0,examples))
    base=model(batch['records'],batch['query'])
    base_loss=per_loss(base,batch['answer'])
    base_refs=torch.stack([x[:,-1] for x in base.operands],1)
    rng=random.Random(830000+seed)
    natural=[]
    for number in (1,2,3):
        stats=dict(intervened_fields=number,trials=0,nonzero=0,operand_changed=0,sum_abs_delta=0.)
        for _ in range(repeats):
            changes={}
            while len(changes)<number:
                key,actions,_=intervention(base,rng)
                changes[key]=actions
            # Descendant policies recompute; untouched fields are not frozen.
            alt=model(batch['records'],batch['query'],overrides=changes)
            delta=base_loss-per_loss(alt,batch['answer'])
            refs=torch.stack([x[:,-1] for x in alt.operands],1)
            stats['trials']+=examples
            stats['nonzero']+=int(delta.abs().gt(1e-7).sum())
            stats['operand_changed']+=int(refs.ne(base_refs).any(-1).any(-1).sum())
            stats['sum_abs_delta']+=float(delta.abs().sum())
        natural.append(stats)
    # Construct a controlled conjunctive path at the final layer. The correct
    # second operand is older than one distractor in its two-record bucket.
    # Wrong writer relation/address OR NULL rank prevents it being selected.
    # Only writer assignments/rank vary: relation embeddings stay fixed, so
    # they cannot manufacture loss changes through a direct operator path.
    eligible=[];distractors=[]
    for row,(first,second) in enumerate(batch['support'].tolist()):
        later=[j for j in range(second+1,16) if j!=first]
        if later:
            eligible.append(row);distractors.append(later[0])
    b=take(batch,torch.tensor(eligible,dtype=torch.long))
    control=model(b['records'],b['query'])
    choices={k:v.clone() for k,v in control.decisions.items()}
    ell=1
    wr,wa=choices[(ell,'write_relation')],choices[(ell,'write_address')]
    wr[:,:-1]=2;wa[:,:-1]=2
    for row,((first,second),distractor) in enumerate(zip(b['support'].tolist(),distractors)):
        wr[row,first]=wa[row,first]=3
        wr[row,second]=wa[row,second]=0
        wr[row,distractor]=wa[row,distractor]=0
    choices[(ell,'routes')][:,-1,0]=3
    choices[(ell,'routes')][:,-1,1]=0
    choices[(ell,'query_first')][:,-1]=3
    choices[(ell,'query_second')][:,-1]=0
    choices[(ell,'rank_first')][:,-1]=0
    choices[(ell,'rank_second')][:,-1]=1
    choices[(ell,'page_first')][:,-1]=0
    choices[(ell,'page_second')][:,-1]=0
    joint=model(b['records'],b['query'],replay=choices)
    assert torch.equal(joint.operands[-1][:,-1],b['support'])
    joint_loss=per_loss(joint,b['answer'])
    controlled=[]
    for number in (1,2,3):
        broken={k:v.clone() for k,v in choices.items()}
        for row,(_,target) in enumerate(b['support'].tolist()):
            broken[(ell,'write_relation')][row,target]=1
            if number>=2:
                broken[(ell,'write_address')][row,target]=1
            if number>=3:
                broken[(ell,'rank_second')][row,-1]=2
        failed=model(b['records'],b['query'],replay=broken)
        assert failed.operands[-1][:,-1,1].eq(-1).all()
        failed_loss=per_loss(failed,b['answer'])
        item=dict(required_repairs=number,eligible_examples=len(eligible),single_repair_trials=0,
                  single_nonzero=0,joint_nonzero=int((failed_loss-joint_loss).abs().gt(1e-7).sum()),
                  joint_sum_abs_delta=float((failed_loss-joint_loss).abs().sum()))
        for field in ('write_relation','write_address','rank_second')[:number]:
            repaired={k:v.clone() for k,v in broken.items()}
            repaired[(ell,field)]=choices[(ell,field)].clone()
            alt=model(b['records'],b['query'],replay=repaired)
            if number>=2:
                assert alt.operands[-1][:,-1,1].eq(-1).all()
                torch.testing.assert_close(alt.logits[:,-1],failed.logits[:,-1],rtol=0,atol=0)
            item['single_repair_trials']+=len(eligible)
            item['single_nonzero']+=int((failed_loss-per_loss(alt,b['answer'])).abs().gt(1e-7).sum())
        controlled.append(item)
    return dict(natural=natural,controlled_conjunction=controlled,
                caveat='Natural counts measure interventions, not required coordination. Controlled paths are deliberately constructed and do not estimate their prevalence in learned routing.')
