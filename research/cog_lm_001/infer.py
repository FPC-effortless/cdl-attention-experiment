"""Generate text, save live document state, or resume generation exactly on CPU."""
import argparse
from pathlib import Path
import torch
from tokenizers import Tokenizer

from model import Successors, restore


def encode_state(state):
    layers,bank,episodes=state
    return dict(layers=layers,bank=bank,episodes=None if episodes is None else
                [dict(capacity=e.capacity,tail=e.tail,entries=list(e.table.items())) for e in episodes])


def decode_state(saved):
    episodes=saved['episodes']
    if episodes is not None:
        restored=[]
        for r in episodes:
            e=Successors(r['capacity']);e.tail=tuple(r['tail']);e.table.update(r['entries']);restored.append(e)
        episodes=restored
    return saved['layers'],saved['bank'],episodes


@torch.inference_mode()
def infer(a):
    import hashlib
    if a.tokens<1 or a.temperature<=0:
        raise ValueError('tokens and temperature must be positive')
    fingerprint=hashlib.file_digest(a.weights.open('rb'),'sha256').hexdigest()
    vocab_hash=hashlib.file_digest(a.tokenizer.open('rb'),'sha256').hexdigest()
    m=restore(a.weights)
    if m.c.arm!='cog':
        raise ValueError('live state export is implemented for COG checkpoints')
    tok=Tokenizer.from_file(str(a.tokenizer))
    rng=torch.Generator().manual_seed(a.seed)
    if a.resume_state:
        s=torch.load(a.resume_state,map_location='cpu',weights_only=True)
        if (fingerprint,vocab_hash)!=(s['weights_sha256'],s['tokenizer_sha256']):
            raise ValueError('checkpoint or tokenizer mismatch')
        if s['ended']:
            raise ValueError('the saved document already reached EOS')
        rng.set_state(s['rng'])
        logp,state=s['logp'],decode_state(s['state'])
    else:
        ids=tok.encode(a.prompt).ids
        if not ids:
            raise ValueError('supply a nonempty prompt')
        logp,state=m(torch.tensor([ids]))
    generated=[]
    for _ in range(a.tokens):
        token=torch.multinomial((logp[:,-1]/a.temperature).softmax(-1),1,generator=rng)
        generated.append(token.item())
        logp,state=m(token,state)
        if token.item()==0:
            break
    if a.save_state:
        torch.save(dict(weights_sha256=fingerprint,tokenizer_sha256=vocab_hash,state=encode_state(state),
                        rng=rng.get_state(),logp=logp,ended=generated[-1]==0),a.save_state)
    print(tok.decode(generated))
    return generated


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--tokenizer',type=Path,required=True)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--prompt');source.add_argument('--resume-state',type=Path)
    p.add_argument('--save-state',type=Path);p.add_argument('--tokens',type=int,default=64)
    p.add_argument('--temperature',type=float,default=.8);p.add_argument('--seed',type=int,default=31)
    args=p.parse_args();torch.set_num_threads(2);infer(args)
