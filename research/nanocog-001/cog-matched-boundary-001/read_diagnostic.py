"""Post-fit, label-aware readout intervention. Never used for qualification."""
import argparse
from pathlib import Path
import torch
from run import ARMS, SEEDS, Model, enc, read, sha, put, y_encode, MODES, req

def test(m,x,target):
    # Capture exactly the native clause values and query; the only substitution
    # in the intervention is the vector sent to the existing final head.
    captured={}
    def value_hook(_m,_i,o):captured['v']=o.chunk(2,-1)[1]
    def q_hook(_m,_i,o):captured['q']=o
    vh=m.kv.register_forward_hook(value_hook); qh=m.q.register_forward_hook(q_hook)
    outputs={k:[] for k in ('native','true_support','wrong_support','uniform')}
    with torch.no_grad():
        try:
            for t,ix in zip(x.split(256),target.split(256),strict=True):
                z,w=m(t);v=captured['v'];q=captured['q']
                batch=torch.arange(len(t))
                req(torch.equal(z,m.final(torch.cat((q,(v*w[:,:,None]).sum(1)),dim=-1))), 'native reconstruction failed')
                outputs['native'].append(z)
                for label,c in [('true_support',v[batch,ix]),('wrong_support',v[batch,(ix+1)%3]),('uniform',v.mean(1))]:
                    outputs[label].append(m.final(torch.cat((q,c),dim=-1)))
        finally:
            vh.remove();qh.remove()
    return {k:torch.cat(a) for k,a in outputs.items()}

def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--results',type=Path,required=True)
    a=ap.parse_args()
    sealed=a.results/'RESULTS.json';req(sealed.exists(),'primary must be scored first')
    out={}
    cohort='novel'
    pp=enc(read(a.data/(cohort+'-prompts.json')))
    truth=read(a.data/(cohort+'-truth.json')); yy=y_encode(truth)
    targets=torch.tensor([d['target'] for d in truth]).flatten()
    for arm in ARMS:
        for seed in SEEDS:
            key=f'{arm}-{seed}';p=a.results/key
            report=read(p/'fit.json');req(sha(p/'weights.pt')==report['weight_sha'],'model custody')
            m=Model(arm);m.load_state_dict(torch.load(p/'weights.pt',map_location='cpu',weights_only=True));m.eval()
            zn=test(m,pp,targets)
            wire=torch.load(p/(cohort+'.pt'),map_location='cpu',weights_only=True)['logits']
            req(torch.allclose(zn['native'],wire,atol=1e-5,rtol=1e-6),'native wire mismatch')
            res={mode:{} for mode in MODES}
            base=None
            for name,z in zn.items():
                top=z.max(-1,keepdim=True).values
                eq=((z.argmax(-1)==yy)&((z==top).sum(-1)==1)).reshape(-1,4)
                if base is None: base=eq
                for mode in MODES:
                    ix=[i for i,t in enumerate(truth) if t['mode']==mode]
                    x=eq[ix];b=base[ix]
                    res[mode][name]={'row':int(x.sum()),'strict':int(x.all(-1).sum()),
                                     'family_rescued':int((x.all(-1)&~b.all(-1)).sum()),
                                     'family_damaged':int((~x.all(-1)&b.all(-1)).sum())}
            out[key]=res
    put(a.results/'READ_DIAGNOSTIC.json',{'primary_sha':sha(sealed),'cohort':cohort,'results':out})
if __name__=='__main__':main()
