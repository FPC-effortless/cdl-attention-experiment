"""Check that regrouping changes only the order of otherwise identical gradients."""
import json
import torch
from train import F, batches, blocks, build_model, ids

def gradient(model,updates):
    model.zero_grad()
    for batch in updates:
        x=ids([text for text,_ in batch])
        y=torch.tensor([value for _,value in batch])
        (F.cross_entropy(model(x)[:,-1],y)/len(updates)).backward()
    return torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])

if __name__=="__main__":
    torch.set_num_threads(1)
    report={}
    for domain in ("symbol","english"):
        model=build_model("C",9731)
        block=next(blocks(9731,domain))
        for i in range(0,8,2):
            if block[i][0][1]==block[i+1][0][1]:
                block[i+1]=block[i+1][2:]+block[i+1][:2]
        grouped=gradient(model,batches(block,True))
        mixed=gradient(model,batches(block,False))
        error=float((grouped-mixed).norm()/grouped.norm())
        assert error<1e-5
        report[domain]=error
    print(json.dumps(report,indent=2))
