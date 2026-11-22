"""Narrow load-only compatibility adapter for reviewed immutable torch.save wires."""
from dataclasses import asdict
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from fit_value_path import BASE,BASE_TENSOR,ARMS,SEEDS,ENDPOINTS,Plan,need,hash_file

SCHEMA='cog-info-value-repair.adapter.v1'
REGISTERED_GLOBAL='torch.torch_version.TorchVersion'
FIELDS={'plan','base_checkpoint_sha256','base_state_sha256','train_manifest_sha256','schedule_sha256',
        'initial_adapter_state_sha256','vocab_size','torch','python','torch_threads','torch_interop_threads',
        'claim_eligibility','witness_sha256','schema','arm','seed','updates','adapter_state_sha256','tokenizer_words','state_dict'}
SHAPES={'first.weight':(8,68),'last.weight':(32,8)}
PINNED_TORCH_AUTO_GLOBALS={'torch.nested._internal.nested_tensor._rebuild_njt',
                            'torch.nested._internal.nested_tensor.NestedTensor',
                            'torch.distributed._mesh_layout._MeshLayout'}

def isolate_allowlist():
    # Torch itself installs these defaults at import. The isolated process
    # needs none of them; discard the exact known roster before any adapter.
    import torch
    ambient=torch.serialization.get_safe_globals()
    ids={f'{x.__module__}.{x.__qualname__}' for x in ambient
         if not isinstance(x,tuple) and hasattr(x,'__qualname__')}
    need((not ambient or (len(ambient)==3 and ids==PINNED_TORCH_AUTO_GLOBALS)),
         'unexpected extra/ambient user-safe global')
    torch.serialization.clear_safe_globals()
    need(not torch.serialization.get_safe_globals(),'unable to isolate adapter deserialization')


def state_sha(state):
    h=sha256()
    for name,t in sorted(state.items()):
        h.update(name.encode());h.update(str((list(t.shape),str(t.dtype))).encode())
        h.update(t.contiguous().numpy().tobytes())
    return h.hexdigest()

def safe_adapter(path,expect,torch_code):
    import torch
    from torch.torch_version import TorchVersion
    need(torch.__version__=='2.10.0+cpu' and
         set(torch_code)=={'torch_version.py','serialization.py'} and
         all(hash_file(Path(torch.__file__).parent/k)==v for k,v in torch_code.items()) and
         type(torch.__version__) is TorchVersion and not torch.serialization.get_safe_globals(),
         'exact TorchVersion source and no ambient allowlist')
    p=Path(path)
    need(p.is_file() and not p.is_symlink() and p.stat().st_size<=65536,'regular bounded adapter')
    raw=p.read_bytes()
    need(sha256(raw).hexdigest()==expect['sha256'],'exact independent checkpoint bytes before archive inspection')
    need(torch.serialization.get_unsafe_globals_in_checkpoint(BytesIO(raw))==[REGISTERED_GLOBAL],
         'unsupported/absent adapter metadata global')
    with torch.serialization.safe_globals([TorchVersion]):
        ad=torch.load(BytesIO(raw),map_location='cpu',weights_only=True)
    need(not torch.serialization.get_safe_globals(),'scoped allowlist not restored')
    arm,seed,u=(expect[k] for k in ('arm','seed','updates'))
    need(type(ad)is dict and set(ad)==FIELDS and
      ad['schema']==SCHEMA and arm in ARMS and seed in SEEDS and u in ENDPOINTS and
      (ad['arm'],ad['seed'],ad['updates'])==(arm,seed,u) and type(ad['seed'])is type(ad['updates'])is int and
      ad['plan']==asdict(Plan(arm,seed)) and ad['base_checkpoint_sha256']==BASE and
      ad['base_state_sha256']==BASE_TENSOR and ad['vocab_size']==287 and
      type(ad['torch']) is TorchVersion and str(ad['torch'])=='2.10.0+cpu' and ad['python'].startswith('3.12.14 ') and
      ad['torch_threads']==ad['torch_interop_threads']==1 and ad['claim_eligibility']=='NOT_QUALIFIED',
      'exact endpoint common metadata')
    for k in ('train_manifest_sha256','schedule_sha256','initial_adapter_state_sha256','witness_sha256',
              'adapter_state_sha256'):
        need(ad[k]==expect[k],f'exact reviewed endpoint provenance: {k}')
    ww=ad['tokenizer_words']
    need(type(ww)is list and len(ww)==len(set(ww))==25 and all(type(w)is str for w in ww),'exact literal vocabulary structure')
    sd=ad['state_dict']
    need(type(sd)is dict and set(sd)==set(SHAPES) and all(type(t)is torch.Tensor
      and tuple(t.shape)==SHAPES[k] and t.dtype==torch.float32 and t.device.type=='cpu' and
      t.is_contiguous() and not t.requires_grad and bool(torch.isfinite(t).all()) for k,t in sd.items()),
      'only exact finite FP32 CPU adapter tensors')
    need(state_sha(sd)==expect['adapter_state_sha256'],'tensor digest matches reviewed receipt')
    return ad
