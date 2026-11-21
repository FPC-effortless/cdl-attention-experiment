"""Sparse causal scoring of the unchanged 002 head on cached frozen FP32 representations."""
import argparse
import time
import math
from common import Path, np, torch, Runtime, sha, save, model_paths, build_docs
from torch.nn import functional as F
from architecture import mix

CONFIGS={
 'short_text':('cog_revision',True,False,False),
 'long_text':('cog_revision',False,False,False),
 'short_revision':('cog_revision',True,True,False),
 'long_revision':('cog_revision',False,True,False),
 'floor_revision':('cog_revision',False,True,True),
 'route_revision':('cog_revision',False,True,True),
 'cog_control':('cog_lookup',False,True,False),
 'route_cog_control':('cog_lookup',False,True,True),
 'mamba_control':('mamba_lookup',False,True,False),
 'route_mamba_control':('mamba_lookup',False,True,True)
}


def sparse(head,emb,h,lookup,selected,short=False,mask=False):
    # The 002 bank only reads into the language output. All its writes use
    # pre-read hidden states. Batched projections preserve the serial recurrence.
    base=h+head.up(F.gelu(head.down(h)))
    z=base[selected]
    bank=head.bank
    ratio=z.new_zeros(len(z))
    if bank is not None:
        complete=base.shape[0]//bank.period
        blocks=base[:complete*bank.period].reshape(complete,bank.period,-1)
        pooled=(bank.salience(blocks).softmax(1)*blocks).sum(1)
        newkeys=F.normalize(bank.key(pooled),dim=-1)
        vals=bank.value(pooled).tanh()
        gains=bank.strength(pooled).sigmoid()
        keys=bank.anchors
        values=torch.zeros_like(keys)
        history_keys=[]
        history_vals=[]
        for block in range(complete+1):
            if short and block%(128//bank.period)==0:
                keys=bank.anchors
                values=torch.zeros_like(keys)
            history_keys.append(keys)
            history_vals.append(values)
            if block<complete:
                route=(8*(F.normalize(keys,dim=-1)@newkeys[block])).softmax(0)
                rate=route[:,None]*gains[block]
                keys=torch.lerp(keys,newkeys[block].expand_as(keys),rate)
                values=torch.lerp(values,vals[block].expand_as(values),rate)
        addr=F.normalize(torch.stack(history_keys)[selected//bank.period],dim=-1)
        mem=torch.stack(history_vals)[selected//bank.period]
        q=F.normalize(bank.query(z),dim=-1)
        weights=(8*(addr*q[:,None]).sum(-1)).softmax(-1)
        readout=bank.readout((weights[:,:,None]*mem).sum(1))
        ratio=readout.norm(dim=-1)/z.norm(dim=-1).clamp_min(1e-9)
        z=z+readout*(0 if mask else 1)
    p=F.log_softmax(F.linear(z,emb),-1)
    return mix(p[None],lookup[selected][None],head.gate(z).squeeze(-1)[None])[0],ratio


def train(a):
    torch.set_num_threads(a.threads)
    arm,short,supervised,pressure=CONFIGS[a.name]
    init,head_path=model_paths(a.parents,arm)
    network=Runtime.load(init,head_path)
    head,emb=network.head,network.model.emb.weight
    docs=build_docs(a.data,'train')
    rng=np.random.default_rng(6083)
    opt=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.01)
    t=time.perf_counter()
    curve=[]
    gradlog=[]
    total=masked=answers=0
    # Only deterministic feature files from the correct 002 arm enter these fits.
    cached=[torch.load(a.cache/(arm+'-train-'+str(i)+'.pt'),weights_only=True)['short' if short else 'long'] for i in range(len(docs))]
    for epoch in range(8):
        for group in rng.permutation(len(docs)//2):
            opt.zero_grad(set_to_none=True)
            for flip in (0,1):
                i=2*int(group)+flip
                doc=docs[i]
                hid,index,queries=cached[i]
                stop=len(hid)
                # Offsets do not depend on arm or on labels. Two flips balance every update.
                positions=np.linspace(len(doc['prefix']),stop-2,8,dtype=int)
                pos=torch.tensor([*positions,0])
                tokens=doc['prefix']+doc['filler']
                qid=(epoch+int(group))%4
                q=doc['questions'][qid]
                drop=pressure and ((epoch//4+int(group))%2==0)
                qh,qi=queries[stop][qid][int(drop)]
                pos[-1]=stop+len(qh)-1
                p,ratio=sparse(head,emb,torch.cat([hid,qh]),torch.cat([index,qi]),pos,short)
                natural=-p[:-1].gather(1,torch.tensor([tokens[j+1] for j in positions])[:,None]).mean()
                answer=-p[-1,q['target']]
                loss=natural+(answer if supervised else 0.)
                if a.name=='floor_revision':
                    shortfall=F.relu(math.log(.01)-torch.log(ratio[-1].clamp_min(1e-9)))
                    loss=loss+.25*shortfall.square()
                (loss/2).backward()
                total+=stop+len(qh)
                masked+=int(drop)
                answers+=int(supervised)
                curve.append([float(natural.detach()),float(answer.detach()),float(ratio[-1].detach())])
            grad=float(torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True))
            gradlog.append(grad)
            opt.step()
        print(a.name,epoch+1,round(time.perf_counter()-t,1),flush=True)
    p=a.out/(a.name+'.pt')
    torch.save(head.state_dict(),p)
    save(a.out/(a.name+'.fit.json'),dict(name=a.name,arm=arm,seed=6083,epochs=8,
        presentations=len(curve),unique_docs=len(docs),source_tokens_presented=total,updates=len(gradlog),
        sampled_lm_targets=len(curve)*8,answer_targets=answers,masked_queries=masked,
        trained_parameters=sum(p.numel() for p in head.parameters()),backbone_sha=sha(init),
        parent_head_sha=sha(head_path),head_sha=sha(p),curve=curve,grad_norm=gradlog,wall_s=time.perf_counter()-t))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('cache','data','parents','out'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--name',choices=CONFIGS,required=True)
    p.add_argument('--threads',type=int,default=1)
    train(p.parse_args())
