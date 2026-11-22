"""Restricted checkpoint decoder shared by inert tests and later blind actors."""
from hashlib import sha256
from io import BytesIO
from joint_local_objective import ARMS,SEEDS,ENDPOINTS,need

NATIVE_FILE='517c990e5834a1648524b59bd287f97f472583fa642ce0e5a8dc95270536e890'
NATIVE_STATE='1a4558037b5bfb046e8fd1a7a3c4265ce35222eda95fb737ba2e40839927d635'
NATIVE_TRAIN='02607c19e760e164989865a54422eb7cd78fed7d117a7fa2ff6c4df9d700f9f8'
WORDS=tuple(sorted('ada ben cy di eli fay gio hal ivy jay kai lou mia nik ola pam blue gold green orange pink purple red white is'.split()))
JOINT_FIELDS={'plan','initial_state_sha256','initial_shared_trunk_sha256','slots_sha256','train_data_pins',
              'tokenizer_words','vocab_size','runtime','status','schema','updates','arm','seed','state_sha256','state_dict'}

def parameter_shapes(mixer):
    out={'embedding.weight':(288,32),'output.weight':(288,32),'resid':(2,),'x0_gain':(2,),
         'smear_gate.weight':(1,24),'smear_gain':(),'backout':()}
    for i in (0,1):
        out[f'mlps.{i}.0.weight']=(128,32);out[f'mlps.{i}.1.weight']=(32,128)
        q=f'mixers.{i}.local.'
        if mixer=='gated32':
            out.update({q+'offset_gate.weight':(4,32),q+'offset_gate.bias':(4,),
                        q+'depthwise':(32,4),q+'pointwise.weight':(32,32)})
        elif mixer=='attention32':
            out.update({q+k+'.weight':(10,32) for k in ('key','query','value')})
            out.update({q+'project.weight':(32,10),q+'relative':(4,)})
        else:raise ValueError('unsupported declared local mixer')
    return out

def state_sha(sd):
    h=sha256()
    for name,t in sorted(sd.items()):
        h.update(name.encode());h.update(str((list(t.shape),str(t.dtype))).encode())
        h.update(t.contiguous().numpy().tobytes())
    return h.hexdigest()

def joint_plan(arm,seed):
    mixer,tail=ARMS[arm]
    return dict(arm=arm,seed=seed,mixer=mixer,tail=tail,lr=.001,warmup=8,weight_decay=.01,
        clip=1.,betas=[.9,.999],eps=1e-8,prefix=.25,margin=.25,tail_weight=.5 if tail else 0.,temp=1.)

def validate_metadata(ad,expect):
    need(type(ad)is dict and ad['tokenizer_words']==list(WORDS) and ad['vocab_size']==287,'static literal vocabulary')
    arm,seed,step=map(expect.get,('arm','seed','updates'))
    if arm=='native':
        need((seed,step)==(1371,0) and expect['state_sha256']==NATIVE_STATE and
            ad['schema']=='cog-comm-role-001.checkpoint.v1' and ad['format']==1 and
            ad['seed']==1371 and ad['updates']==1280 and ad['train_sha256']==NATIVE_TRAIN and
            ad['plan']['arm']=='gated32' and ad['plan']['clip']==1. and ad['plan']['prefix_ce_weight']==.25,
            'exact old fixed native comparator metadata')
        return 'gated32'
    need(arm in ARMS and seed in SEEDS and step in ENDPOINTS and
         set(ad)==JOINT_FIELDS and ad['schema']=='cog-joint-local-001.checkpoint.v1' and ad['status']=='NOT_QUALIFIED'
         and (ad['arm'],ad['seed'],ad['updates'])==(arm,seed,step)
         and type(ad['seed']) is type(ad['updates']) is int and ad['plan']==joint_plan(arm,seed),
         'fixed exact joint-endpoint metadata')
    for k in ('initial_state_sha256','initial_shared_trunk_sha256','slots_sha256','train_data_pins','runtime','state_sha256'):
        need(ad[k]==expect[k],'independent FIT metadata mismatch: '+k)
    need(type(ad['runtime']['torch_version']) is str and ad['runtime']['torch_version']=='2.10.0+cpu',
         'plain string metadata only, no TorchVersion object')
    return ARMS[arm][0]

def decode_raw(raw,expect,*,torch):
    need(0<len(raw)<=512_000 and sha256(raw).hexdigest()==expect['sha256'],'bound immutable artifact before scanning')
    if expect['arm']=='native':
        need(expect['sha256']==NATIVE_FILE,'registered native checkpoint only')
    need(torch.serialization.get_unsafe_globals_in_checkpoint(BytesIO(raw))==[],
         'unexpected global: no allowlist, fallback or custom metadata classes')
    ad=torch.load(BytesIO(raw),map_location='cpu',weights_only=True)
    mixer=validate_metadata(ad,expect);shapes=parameter_shapes(mixer)
    sd=ad['state_dict']
    need(type(sd)is dict and set(sd)==set(shapes) and all(type(t)is torch.Tensor and tuple(t.shape)==shapes[n] and
         t.dtype==torch.float32 and t.device.type=='cpu' and t.is_contiguous() and
         not t.requires_grad and bool(torch.isfinite(t).all()) for n,t in sd.items()),'exact safe model-only state tensors')
    need(sum(p.numel() for p in sd.values())==37414 and state_sha(sd)==expect['state_sha256'],'independently pinned decoded state')
    return mixer,sd
