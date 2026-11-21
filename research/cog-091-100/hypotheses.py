"""COG-ATTN-095: continuous alternatives before symbolic verification.

A shallow multi-label hypothesis readout shares the same parallel convolution
features as the action model, but predicts a separate distribution over possible
intended actions. Its outputs are *provisional*, never authorization.
"""
import argparse,json
import torch
from torch import nn
from semantic import Router,SEEDS,make_data,to_tensor

class HypothesisNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=Router('parallel_specialists')
        self.features.out=nn.Identity()
        self.action_head=nn.Linear(72,7)
        self.candidate_head=nn.Linear(72,6)

    def forward(self,x,lengths):
        h=self.features(x,lengths)
        return self.action_head(h),self.candidate_head(h)


def targets(rows):
    multi=torch.zeros(len(rows),6)
    for i,r in enumerate(rows):
        if r['answer']==6:
            a,b=r['pair'];multi[i,a]=multi[i,b]=1
        else:multi[i,r['answer']]=1
    return multi


def score(model,raw,arr):
    model.eval()
    with torch.no_grad():
        a,h=model(arr[0],arr[1]);pred=a.argmax(1)
        ambiguous=[i for i,r in enumerate(raw) if r['answer']==6]
        top=h.topk(2,dim=1).indices
        support=sum(set(top[i].tolist())==set(raw[i]['pair']) for i in ambiguous)
        defer=sum(int(pred[i]==6) for i in ambiguous)
        # After a later explicit clarification selects one of the two alternatives,
        # downstream verified execution may use it only if the proposed action is
        # still inside the retained candidate set. No COG write is performed here.
        clarified_success=sum(raw[i]['pair'][0] in set(top[i].tolist()) and
                              raw[i]['pair'][1] in set(top[i].tolist()) for i in ambiguous)
        return dict(correct=int((pred==arr[2]).sum()),n=len(raw),
                    ambiguous_n=len(ambiguous),pair_top2=int(support),
                    defer_on_ambiguity=int(defer),clarification_recoverable=int(clarified_success))


def run_one(seed,epochs=38):
    torch.manual_seed(seed)
    d=make_data(seed);cache={};train=to_tensor(d['train'],cache);dev=to_tensor(d['dev'],cache);test=to_tensor(d['test'],cache)
    multi=targets(d['train']);model=HypothesisNet()
    optim=torch.optim.AdamW(model.parameters(),lr=0.007,weight_decay=0.0001)
    best=-1;chosen=None
    for epoch in range(epochs):
        model.train();order=torch.randperm(len(d['train']))
        for ix in order.split(48):
            a,h=model(train[0][ix],train[1][ix])
            loss=nn.functional.cross_entropy(a,train[2][ix])+0.5*nn.functional.binary_cross_entropy_with_logits(h,multi[ix])
            optim.zero_grad();loss.backward();optim.step()
        devscore=score(model,d['dev'],dev)
        key=devscore['correct']/devscore['n']+devscore['pair_top2']/devscore['ambiguous_n']
        if key>best+1e-9:
            best=key;chosen=(epoch+1,{k:v.detach().clone() for k,v in model.state_dict().items()})
    model.load_state_dict(chosen[1]);return dict(seed=seed,selected_epoch=chosen[0],dev_composite=best,**score(model,d['test'],test))


def run():
    torch.set_num_threads(1)
    rows=[run_one(s) for s in SEEDS]
    return dict(name='COG-ATTN-095 bounded multi-hypothesis readout',rows=rows,
                limitations=['Candidate action space is known and fixed at six',
                             'Outputs trained with two-hot supervisory targets for ambiguous phrases',
                             'Clarification recoverability is candidate coverage only; no interactive natural dialogue was tested',
                             'No external authority, signed token, real action or full COG controller was invoked'])

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='hypotheses-results.json');args=ap.parse_args()
    result=run();open(args.out,'w').write(json.dumps(result,sort_keys=True,indent=2)+'\n')
    for row in result['rows']:print(row)
