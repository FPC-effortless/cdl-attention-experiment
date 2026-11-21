from pathlib import Path
import sys
import hashlib
import json

import numpy as np
import torch
from tokenizers import Tokenizer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'cog_lm_002'))
from architecture import Readout, backbone, features, successor_ids, Successors
from runtime import Runtime
from data import corpus


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text())


def save(path, data):
    path.write_text(json.dumps(data)+'\n')


def model_paths(parents, arm):
    tag='joint_'+arm
    base=parents/(tag+'.base.pt')
    head=parents/(tag+'.pt')
    meta=read(parents/(tag+'.json'))
    expected=read(Path(__file__).resolve().parents[1]/'cog_lm_002/evidence/summary.json')['fits'][tag]
    for path,key in ((base,'base_sha256'),(head,'adapter_sha256')):
        if sha(path)!=expected[key] or meta[key]!=expected[key]:
            raise ValueError('002 checkpoint lineage mismatch')
    return base,head


def build_docs(data,split):
    tok=Tokenizer.from_file(str(data/'tokenizer.json'))
    encode=lambda s:tok.encode(s).ids
    ids,offs=corpus(data,'train' if split=='train' else 'test')
    train=split=='train'
    pairs=[('Anna','David'),('Robert','Mary'),('Jack','Peter'),('Helen','Ruth'),
           ('Susan','Alice'),('Paul','James'),('Jane','Henry'),('Mark','Sarah')] if train else [
           ('Mira','Owen'),('Nora','Iris'),('Felix','Lena'),('Amir','Zara')]
    alternatives=[('north','south'),('east','west')]
    rng=np.random.default_rng(492 if train else 831)
    delays=(64,512,2048) if train else (256,1024,4096,16384)
    docs=[]
    for family,(first,second) in enumerate(pairs):
        for block,d in enumerate(delays if train else (max(delays),)):
            eligible=np.flatnonzero(np.diff(offs)>d+128)
            article=int(rng.choice(eligible))
            start=int(rng.integers(offs[article]+16,offs[article+1]-d-1))
            filler=ids[start:start+d].astype('int64').tolist()
            a,b=alternatives[(family+block)%2]
            for flip in (0,1):
                v,w=(a,b) if flip==0 else (b,a)
                # After update, both people have opposite current values. Balance whether
                # the queried person was revised by issuing both questions separately.
                clause='The direction for '+first+' is'
                clause2='The direction for '+second+' is'
                pre=clause+' '+w+'. '+clause2+' '+w+'. '
                if train:
                    pre+=['Change of plan. '+first+' will now head '+v+'.',
                          'Correction for '+first+': the revised route goes '+v+'.',
                          first+' has switched course and will travel '+v+'.'][(family+block)%3]
                else:
                    pre+='An amendment has arrived: '+first+' is to proceed '+v+' instead.'
                doc=dict(family=family,flip=flip,article=article,start=start,delay=d,prefix=encode(pre),
                         filler=filler,questions=[])
                for who,name,new,old in [('changed',first,v,w),('unchanged',second,w,v)]:
                    literal=' The direction for '+name+' is'
                    paraphrase=([' Where will '+name+' travel? The route leads',
                        ' Which way is '+name+' headed? The answer is'][(family+block)%2] if train else
                        [' Which bearing applies to '+name+' now? It is',
                        ' According to the updated itinerary, '+name+' should go'][family%2])
                    for kind,query in [('literal',literal),('reword',paraphrase)]:
                        target,=encode(' '+new)
                        foil,=encode(' '+old)
                        doc['questions'].append(dict(who=who,kind=kind,q=encode(query),target=target,foil=foil))
                docs.append(doc)
    return docs


@torch.no_grad()
def sequence(base, tokens, state=None):
    # Pad a final short tail only on disposable runs; return state exactly at the true end.
    x=torch.tensor(tokens,dtype=torch.long)[None]
    if base.c.arm=='mamba2' and state is None and len(tokens)<16:
        from architecture import InferenceCache
        state=[InferenceCache.alloc(1,b.mcfg,device=x.device) for b in base.blocks]
    output=[]
    for part in x.split(256,dim=1):
        pieces=[part]
        if base.c.arm=='mamba2' and part.size(1)%16:
            cut=part.size(1)//16*16
            pieces=([part[:,:cut]] if cut else [])+list(part[:,cut:].split(1,dim=1))
        for piece in pieces:
            h,state=features(base,piece,state)
            output.append(h[0])
    return torch.cat(output),state
