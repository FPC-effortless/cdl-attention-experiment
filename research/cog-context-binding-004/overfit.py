"""Can the architecture fit even one four-answer symbolic grid from ordinary bytes?"""
import json, sys
from experiment import build_model, ContextBlock, cases, ids, torch, F, Path

def data(rows):
    x=ids([h+q for a,b,_,_ in rows for h in (a,b) for q in ("#a00=","#b00=")])
    y=torch.tensor([48+k for _,_,a,b in rows for k in (a,b,b,a)])
    return x,y

def fit(arm):
    torch.set_num_threads(1)
    m=build_model(arm if arm in ("GQA","GRU") else "C",9731)
    if arm in ("write","read","both"):
        for block in m.parent.core.blocks:
            block.__class__=ContextBlock
            block.context_write=arm in ("both","write")
            block.context_read=arm in ("both","read")
    training=list(cases(1,861359,False,"symbol",independent=True))
    held=list(cases(64,957427,True,"symbol",independent=True))
    x,y=data(training)
    opt=torch.optim.AdamW(m.parameters(),lr=.0008,betas=(.9,.95),weight_decay=.01)
    stable=0
    for step in range(1,257):
        m.train();opt.zero_grad()
        loss=F.cross_entropy(m(x)[:,-1],y)
        loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
        opt.step();m.eval()
        with torch.no_grad():
            fit_count=int((m(x)[:,-1].argmax(-1)==y).sum())
        stable=stable+1 if fit_count==4 else 0
        if stable==5:break
    correct=0
    with torch.no_grad():
        for i in range(0,len(held),8):
            xs,ys=data(held[i:i+8])
            correct+=int((m(xs)[:,-1].argmax(-1)==ys).reshape(-1,4).all(1).sum())
    return dict(arm=arm,seed=9731,steps=step,training_answers_correct=fit_count,consecutive_perfect=stable,held_grids=correct,held_total=len(held))

if __name__=="__main__":
    arm=sys.argv[1]
    Path(sys.argv[2]).write_text(json.dumps(fit(arm),indent=2))
