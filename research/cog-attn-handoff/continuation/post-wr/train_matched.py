import json,sys,time
from pathlib import Path
root=Path(__file__).resolve().parent/'recipe'
sys.path.insert(0,str(root))
from attention import Decoder, PrefixDecoder,torch
from grounding import dataset,features,np
torch.set_num_threads(1)
reports=[]
for seed in (13,37,61):
    torch.manual_seed(seed)
    spec=json.loads((root/f'data-32-{seed}.json').read_text())
    vocab=spec['vocabulary']
    alias=np.array(spec['aliases'])
    with np.load(root/f'analytic-32-{seed}.npz') as ck:
        teacher=np.concatenate([ck[k] for k in ck.files],axis=1)
    def batch(s,n,c):
        raw,_,_,gold=dataset(s,n,32,alias,c)
        ids=features(raw,vocab)[0]
        logits=teacher[ids].sum(1)
        return torch.tensor(np.pad(ids,((0,0),(0,1)),constant_values=len(vocab)+1)),torch.tensor(logits,dtype=torch.float32),gold
    x,y,_=batch(seed+1000,4096,'familiar')
    model=Decoder(len(vocab))
    with torch.no_grad():
        width = model.embedding.embedding_dim
        ident = torch.eye(width)
        model.embedding.weight.mul_(.001)
        model.embedding.weight[:len(teacher),:66].copy_(torch.tensor(teacher))
        model.embedding.weight[-1].zero_()
        model.qkv.weight[:2*width].mul_(.01)
        model.qkv.bias.zero_()
        model.qkv.weight[2*width:].copy_(ident)
        model.project.weight.copy_(ident)
        model.project.bias.zero_()
        model.ff[-1].weight.mul_(.0001)
        model.ff[-1].bias.zero_()
        model.output.weight.copy_(ident[:66]*x.shape[1])
        model.output.bias.zero_()
    opt=torch.optim.AdamW(model.parameters(),lr=1e-5)
    start=time.perf_counter()
    for step in range(300):
        idx=torch.randint(len(x),(32,))
        out,_=model(x[idx])
        loss=(out-y[idx]).square().mean()
        opt.zero_grad();loss.backward();opt.step()
    np.savez_compressed(root/f'matched-decoder-{seed}.npz',**{k:v.detach().numpy() for k,v in model.state_dict().items()})
    kv=PrefixDecoder(model.eval())
    for i,c in enumerate(('familiar','held_pair','recombine','novel_verb')):
        tx,ty,gold=batch(seed+52000+i,1024,c)
        with torch.no_grad():
            out=torch.cat([model(part)[0] for part in tx.split(128)])
        def pick(logits):
            s,d,o=logits.split([32,32,2],dim=-1)
            return (s.argmax(1)*32+d.argmax(1)+1024*o.argmax(1)).numpy()
        error=max(float((kv(row[:-1].tolist())[0]-expected).abs().max()) for row,expected in zip(tx[:32],out[:32]))
        assert error<1e-4,error
        reports.append(dict(seed=seed,cohort=c,n=len(gold),teacher=int((pick(ty)==gold).sum()),
                            student=int((pick(out)==gold).sum()),max_kv_error=error))
    print(seed,time.perf_counter()-start,flush=True)
    (root.parent/'matched-validation.json').write_text(json.dumps(reports,indent=2))
