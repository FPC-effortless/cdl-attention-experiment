"""022 sentence-parallel learned parser. Not native CSM-equivalent computation.
Exact execution/IF decisions are external ordered typed semantics, not these logits.
"""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

OPS=('NOOP','SET','COPY','LINK','DELETE','QUERY','IFSET')
FIELDS=('dst','src','value','expected')
REQUIRED={'NOOP':(), 'SET':('dst','value'), 'COPY':('dst','src'), 'LINK':('dst','src'),
          'DELETE':('dst',), 'QUERY':('dst',), 'IFSET':('dst','src','value','expected')}
PAD=256

@dataclass(frozen=True)
class Config:
    arm:str='parallel'
    width:int=64
    gru_width:int=72
    max_length:int=256
    heads:int=4


def tensorize_texts(texts,device='cpu',max_length=256):
    if not texts:raise ValueError('empty batch')
    data=[s.encode('ascii') for s in texts]
    if any(not x or len(x)>max_length for x in data):raise ValueError('empty/over-capacity sentence')
    x=torch.full((len(data),max_length),PAD,dtype=torch.long,device=device)
    for i,row in enumerate(data):x[i,:len(row)]=torch.tensor(list(row),device=device)
    return x,torch.tensor([len(r) for r in data],device=device)


class SentenceParser(nn.Module):
    def __init__(self,config=Config()):
        super().__init__();self.config=config
        if config.arm not in ('parallel','gru'):raise ValueError('unknown parser arm')
        w=config.width
        self.embedding=nn.Embedding(257,w,padding_idx=PAD)
        self.positions=nn.Embedding(config.max_length,w)
        if config.arm=='parallel':
            self.encoder=nn.TransformerEncoderLayer(w,config.heads,2*w,dropout=0.,batch_first=True,norm_first=True,activation='gelu')
        else:
            self.encoder=nn.GRU(w,config.gru_width,batch_first=True)
            self.gru_projection=nn.Linear(config.gru_width,w)
        self.norm=nn.LayerNorm(w)
        self.op=nn.Linear(w,len(OPS))
        self.role=nn.ModuleDict({f'{f}_{end}':nn.Linear(2*w,1) for f in FIELDS for end in ('start','end')})
    def forward(self,x,lengths):
        if x.ndim!=2 or lengths.shape!=(x.shape[0],) or x.shape[1]>self.config.max_length:raise ValueError('shape mismatch')
        if bool(((lengths<=0)|(lengths>x.shape[1])).any()):raise ValueError('invalid lengths')
        mask=torch.arange(x.shape[1],device=x.device)[None]<lengths[:,None]
        h=self.embedding(x)+self.positions(torch.arange(x.shape[1],device=x.device))[None]
        if self.config.arm=='parallel':
            h=self.encoder(h,src_key_padding_mask=~mask)
            summary=(h*mask[:,:,None]).sum(1)/lengths[:,None]
        else:
            # GRU is sequential inside sentence. No context crosses sentence boundaries.
            h,_=self.encoder(h);h=self.gru_projection(h)
            summary=h[torch.arange(x.shape[0],device=x.device),lengths-1]
        h=self.norm(h);summary=self.norm(summary)
        result={'op':self.op(summary)}
        result.update(self.role_logits(h,summary,mask))
        return result

    def role_logits(self,h,summary,mask):
        # A concatenated additive global term cancels across position logits.
        # Multiplicative local/global features make relative span scores depend
        # on the complete-sentence summary, including later GRU token context.
        role_input=torch.cat((h,h*summary[:,None]),-1)
        return {name:head(role_input).squeeze(-1).masked_fill(~mask,-float('inf'))
                for name,head in self.role.items()}


def supervised_loss(logits,labels):
    """Explicit op/operand-span supervision. Not end-to-end answer-only learning.
    labels: list {'op':integer,'spans':field->[start,end-exclusive], absent[-1,-1]}.
    No soft loss on unused spanheads. Missing REQUIRED annotations reject training.
    """
    device=logits['op'].device
    op=torch.tensor([q['op'] for q in labels],device=device)
    loss=F.cross_entropy(logits['op'],op)
    terms=1
    for f in FIELDS:
        selected=[i for i,q in enumerate(labels) if f in REQUIRED[OPS[q['op']]]]
        if not selected:continue
        for endpoint in ('start','end'):
            targets=[]
            for i in selected:
                a,b=labels[i]['spans'][f]
                if not 0<=a<b<=logits[f'{f}_start'].shape[1]:raise ValueError('invalid required annotation')
                targets.append(a if endpoint=='start' else b-1)
            loss=loss+F.cross_entropy(logits[f'{f}_{endpoint}'][selected],torch.tensor(targets,device=device));terms+=1
    return loss/terms


def decode_logits(logits,lengths):
    """Fail closed on invalid required span; never consult gold roles/operation.
    No automatic source/destination swap, trim, token snap or span repair.
    """
    if any(torch.isnan(v).any() or torch.isposinf(v).any() for v in logits.values()):raise ValueError('nonfinite parser logits')
    if not torch.isfinite(logits['op']).all():raise ValueError('nonfinite opcode logits')
    result=[]
    for i,n in enumerate(lengths.tolist()):
        op=OPS[int(logits['op'][i].argmax())];r={'op':op,'valid':True,'spans':{f:None for f in FIELDS}}
        for f in REQUIRED[op]:
            a=int(logits[f'{f}_start'][i].argmax());b=int(logits[f'{f}_end'][i].argmax())+1
            if not 0<=a<b<=n:r['valid']=False
            r['spans'][f]=[a,b]
        result.append(r)
    return result


@torch.no_grad()
def infer_chunks(model,texts,chunk_size):
    if model.training:raise ValueError('eval mode required')
    if chunk_size not in (1,8,32,128):raise ValueError('declared chunk sizes1,8,32,128 only')
    device=next(model.parameters()).device;all_logits={};actions=[]
    for i in range(0,len(texts),chunk_size):
        x,lengths=tensorize_texts(texts[i:i+chunk_size],device,model.config.max_length)
        z=model(x,lengths);actions.extend(decode_logits(z,lengths))
        for name,t in z.items():all_logits.setdefault(name,[]).append(t.cpu())
    return {k:torch.cat(v) for k,v in all_logits.items()},actions


def parity_guard(reference_logits,reference_actions,logits,actions,atol=1e-5,rtol=1e-5):
    if reference_actions!=actions:raise RuntimeError('chunk argmax/action divergence; timing not qualified')
    max_delta=0.
    for k,a in reference_logits.items():
        b=logits[k]
        if torch.isnan(a).any() or torch.isnan(b).any() or torch.isposinf(a).any() or torch.isposinf(b).any():raise RuntimeError('nonfinite parity logits')
        if not torch.equal(torch.isneginf(a),torch.isneginf(b)):raise RuntimeError('padding support changed')
        valid=torch.isfinite(a)
        if not torch.allclose(a[valid],b[valid],atol=atol,rtol=rtol):raise RuntimeError('chunk logits exceed tolerance')
        if valid.any():max_delta=max(max_delta,float((a[valid]-b[valid]).abs().max()))
    return {'same_actions':True,'max_logit_difference':max_delta,'atol':atol,'rtol':rtol}
