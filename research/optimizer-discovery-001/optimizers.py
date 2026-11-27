"""COG optimizer recipes. All parameters assigned exactly once."""
import math, torch
RECIPES=('adamw','muon','cog_split','soap','schedulefree')
RATES={r:(.0003,.0006,.0012) for r in RECIPES}; RATES['schedulefree']=(.0006,.0012,.0024)
def make_bundle(model,recipe,rate,steps,wd=.1):
    named=list(model.named_parameters()); hidden=[(n,p) for n,p in named if p.ndim==2 and n!='core.emb.weight']
    chosen=hidden if recipe in ('muon','soap') else [(n,p) for n,p in hidden if '.ff.' in n or (n.startswith('core.blocks.') and n.endswith('.out.weight'))] if recipe=='cog_split' else []
    ids={id(p) for n,p in chosen}; rest=[(n,p) for n,p in named if id(p) not in ids]
    groups=[dict(params=[p for n,p in rest if p.ndim>=2],weight_decay=wd),dict(params=[p for n,p in rest if p.ndim<2],weight_decay=0.)]; groups=[g for g in groups if g['params']]
    if recipe=='schedulefree':
        from adamw_schedulefree import AdamWScheduleFree
        ops=[AdamWScheduleFree(groups,lr=rate,betas=(.9,.95),warmup_steps=max(1,round(.02*steps)))]
    else:ops=[torch.optim.AdamW(groups,lr=rate,betas=(.9,.95))]
    if recipe in ('muon','cog_split'):ops.append(torch.optim.Muon([p for n,p in chosen],lr=rate,weight_decay=wd,momentum=.95,adjust_lr_fn='match_rms_adamw'))
    if recipe=='soap':
        from soap import SOAP
        ops.append(SOAP([p for n,p in chosen],lr=rate,weight_decay=wd,betas=(.95,.95),precondition_frequency=10,max_precond_dim=256,merge_dims=False))
    all_ids=[id(p) for o in ops for g in o.param_groups for p in g['params']];assert len(all_ids)==len(set(all_ids))==len(named) and set(all_ids)=={id(p) for n,p in named}
    return Bundle(ops,recipe,rate,steps,dict(matrix_names=[n for n,p in chosen],matrix_params=sum(p.numel() for n,p in chosen),other_params=sum(p.numel() for n,p in rest)))
class Bundle:
    def __init__(self,ops,recipe,rate,steps,partition):self.ops=ops;self.recipe=recipe;self.rate=rate;self.steps=steps;self.partition=partition
    def train(self):
        if self.recipe=='schedulefree':self.ops[0].train()
    def eval(self):
        if self.recipe=='schedulefree':self.ops[0].eval()
    def zero_grad(self):
        for o in self.ops:o.zero_grad(set_to_none=True)
    def step(self,i):
        if self.recipe!='schedulefree':
            w=max(1,round(.02*self.steps));r=self.rate*(i+1)/w if i<w else self.rate*(.1+.45*(1+math.cos(math.pi*(i-w)/max(1,self.steps-1-w))))
            for o in self.ops:
                for g in o.param_groups:g['lr']=r
        for o in self.ops:o.step()
    def state_dict(self):return [o.state_dict() for o in self.ops]
    def load_state_dict(self,states):
        assert len(states)==len(self.ops)
        for o,s in zip(self.ops,states):o.load_state_dict(s)
    def state_bytes(self):
        def size(x):
            if torch.is_tensor(x):return x.numel()*x.element_size()
            if isinstance(x,dict):return sum(size(v) for v in x.values())
            if isinstance(x,(tuple,list)):return sum(size(v) for v in x)
            return 0
        return sum(size(o.state) for o in self.ops)
