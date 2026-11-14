import torch
from model_source import objective

def forward_with_root(model,batch,roots,mutation=None):
    original=model.step;capture={};bi=torch.arange(len(roots),device=roots.device)
    def step(token,state):
        logits,after=original(token,state)
        if model.position==17:capture['root']=after.values[bi,roots]
        return logits,after
    model.step=step
    try:logits,cap=model.sequence(batch['ids'],collect=True,mutation=mutation)
    finally:model.step=original
    if 'root' not in capture:raise ValueError('missing SET capture')
    return logits,cap,capture['root']

def loss_parts(logits,cap,states,batch,weight):
    main,parts=objective(logits,cap,batch)
    penalty=(states[0::2]-states[1::2]).square().mean()
    return main+weight*penalty,{**parts,'main':float(main.detach()),'invariance':float(penalty.detach()),'state_rms':float(states.detach().square().mean().sqrt())}
