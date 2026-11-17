"""Pinned XSum NCM feasibility only. No bundled dataset, unrestricted pickle, or training.
Run with local predownloaded GPT2 tokenizer. Output is summarization, NOT CSM evidence.
"""
import time
PROCESS_START = time.perf_counter()
import argparse, collections, hashlib, json, pathlib, statistics, sys, os, platform, subprocess
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.dont_write_bytecode = True
PINS = {'generator.pt':'f152f390253de886e46303654f01b9e7a2586e609ba3d6b5e86d5969903b1fd4','tokenizer.pt':'1f806ddaf8e023d1933e95efeabcaaf99f795576c0a05045f1a0f9389ab34fa7','config.yaml':'5dee6b7339645a22362f695fbadb27df6e5254db2d3be0bff8516138d95da493'}
FIXTURE = 'The village library announced on Monday that its opening time will change next month. The library will open at nine in the morning instead of ten. Closing time will remain six in the evening. Librarian Ada Mensah said the change will give students an extra hour to study before classes. The announcement does not change weekend hours.'
DIAGNOSTIC_PANEL = [{'id': 'fresh-name-id-01', 'text': 'Nerava Kelm received parcel Q7M482 containing 37 books.', 'fields': {'name': 'Nerava Kelm', 'identifier': 'Q7M482', 'number': '37'}}, {'id': 'fresh-name-id-02', 'text': 'Taviren Sola logged shipment R2K619 with 84 lamps.', 'fields': {'name': 'Taviren Sola', 'identifier': 'R2K619', 'number': '84'}}, {'id': 'fresh-name-id-03', 'text': 'Mavelo Dren recorded order Z8P753 for 26 chairs.', 'fields': {'name': 'Mavelo Dren', 'identifier': 'Z8P753', 'number': '26'}}]
def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def contained_file(root, relative):
 root = root.absolute()
 q = pathlib.PurePosixPath(relative)
 assert not q.is_absolute() and '..' not in q.parts
 path = root.joinpath(*q.parts)
 assert not root.is_symlink() and not any(v.is_symlink() for v in [path, *path.parents])
 assert path.is_file() and path.resolve().is_relative_to(root.resolve())
 return path
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--checkpoints',required=True);ap.add_argument('--text-tokenizer',required=True);ap.add_argument('--device',default='cuda');ap.add_argument('--out',required=True);ap.add_argument('--repeats',type=int,default=3);ap.add_argument('--load-only',action='store_true');a=ap.parse_args()
 assert a.repeats == 3, 'Admitted protocol fixes exactly 3 warm repeats'
 assert a.load_only or a.device in ('cuda','cuda:0'), 'No CPU fallback for admitted execution'
 costs={}
 stage=time.perf_counter()
 source=pathlib.Path(a.source); ck=pathlib.Path(a.checkpoints)
 assert (source/'SOURCE_COMMIT').read_text().strip()=='c70274273f489abc73832b2b98584a8483b6b203'
 assert digest(source/'RUNTIME_MANIFEST.json')=='e17e90642d0cbe9de3b599dd83ac74c60e079a5eee8fd6cc95a9724cfec3b19a'
 entries=json.loads((source/'RUNTIME_MANIFEST.json').read_text())
 allowed={e['path'] for e in entries} | {'SOURCE_COMMIT','SOURCE_MANIFEST.json','RUNTIME_MANIFEST.json'}
 for path in source.rglob('*'):
  assert not path.is_symlink(), 'Symlink in source'
  if path.is_file():assert path.relative_to(source).as_posix() in allowed, 'Unlisted source overlay: '+str(path)
 for entry in entries:
  assert digest(contained_file(source,entry['path']))==entry['sha256'], 'Source hash mismatch: '+entry['path']
 for name,h in PINS.items():assert digest(contained_file(ck,name))==h, name+' digest mismatch'
 td=pathlib.Path(a.text_tokenizer)
 assert digest(contained_file(td,'MANIFEST.json'))=='0dc013b26d35fcdbed190cd7c710d2eff89c687e628a94856ddb10e585cbcb0f'
 tm=json.loads((td/'MANIFEST.json').read_text())
 assert tm['revision']=='607a30d783dfa663caf39e06633721c8d4cfcd7e'
 assert {q.name for q in td.iterdir()}==set(tm['files'])|{'MANIFEST.json'}
 for name,entry in tm['files'].items():
  path=contained_file(td,name);assert path.stat().st_size==entry['bytes'] and digest(path)==entry['sha256']
 costs['hash_and_allowlist_verification_seconds']=time.perf_counter()-stage
 stage=time.perf_counter()
 sys.path.insert(0,str(source))
 import torch
 from omegaconf import OmegaConf
 from models.vqvae import VQVAE
 from models.next_concept import NextConceptTransformer
 from transformers import AutoTokenizer
 costs['framework_and_model_import_seconds']=time.perf_counter()-stage
 torch.manual_seed(1729);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 if not a.load_only:
  assert torch.cuda.is_available(), 'CUDA unavailable; fail closed'
  assert 'T4' in torch.cuda.get_device_name(0), 'Admitted hardware is T4'
  torch.cuda.reset_peak_memory_stats()
 stage=time.perf_counter()
 # The exact hashed config is trusted data, not an arbitrary hydra target.
 cfg=OmegaConf.load(ck/'config.yaml')
 assert cfg.tokenizer.vqvae.vector_quantizer_config.clss=='multiscale_residual_vector_quantizer'
 assert cfg.tokenizer.pre_tokenizer.name=='hf' and cfg.tokenizer.pre_tokenizer.model_id=='gpt2'
 assert not cfg.model.ctx_encoder.enabled and not cfg.model.use_flex_attention
 costs['config_construction_seconds']=time.perf_counter()-stage
 def load(p):
  o=torch.load(p,map_location='cpu',weights_only=True)
  assert type(o) is dict and 'model' in o
  sd=o['model'];assert type(sd) in (dict,collections.OrderedDict)
  assert all(type(k) is str and type(v) is torch.Tensor for k,v in sd.items())
  assert all(torch.isfinite(v).all().item() for v in sd.values() if v.is_floating_point())
  return o
 start=time.perf_counter();tokck=load(ck/'tokenizer.pt');genck=load(ck/'generator.pt')
 vae=VQVAE(**cfg.tokenizer.vqvae);vae.load_state_dict(tokck['model'],strict=True);del tokck
 net=NextConceptTransformer(vqvae_local=vae,config=cfg.model);net.load_state_dict(genck['model'],strict=True)
 allowance=genck['inference']['token_unigram'];assert type(allowance) is torch.Tensor and torch.isfinite(allowance).all();del genck
 # Count physical storages once as well as exposed parameter objects.
 def storage_counts(module):
  d={}
  for v in module.parameters():
   st=v.untyped_storage();d[(str(v.device),st.data_ptr())]=st.nbytes()
  return d
 ng=storage_counts(net);nv=storage_counts(vae);shared=set(ng)&set(nv)
 unique_bytes=sum({**ng,**nv}.values())
 result={'scope':'own synthetic DEV fixture; NCM summarization feasibility, not CSM or QA qualification','source_commit':(source/'SOURCE_COMMIT').read_text().strip(),'checkpoint_revision':'d342ce8aaea58830eef3b8ad473664673e41f692','hashes':PINS,'torch':torch.__version__,'precision':'float32','weights_only':True,'strict_state_dict':True,'load_seconds':time.perf_counter()-start,'generator_parameters':sum(x.numel() for x in net.parameters()),'tokenizer_parameters':sum(x.numel() for x in vae.parameters()),'levels':list(vae.vq.levels),'fixture':FIXTURE}
 result.update(unique_parameter_storage_bytes=unique_bytes,unique_parameter_storage_float32_elements=unique_bytes//4,shared_parameter_storage_bytes=sum(ng[k] for k in shared),generator_exclusive_parameter_storage_bytes=sum(v for k,v in ng.items() if k not in shared),tokenizer_exclusive_parameter_storage_bytes=sum(v for k,v in nv.items() if k not in shared),runtime_source_manifest_sha256='e17e90642d0cbe9de3b599dd83ac74c60e079a5eee8fd6cc95a9724cfec3b19a')
 costs['checkpoint_cpu_load_construction_seconds']=result['load_seconds']
 def unique_tensor_storages(tensors):
  found={}
  for v in tensors:
   st=v.untyped_storage();key=(str(v.device),st.data_ptr())
   if key not in found:found[key]={'bytes':st.nbytes(),'elements':st.nbytes()//v.element_size(),'dtype':str(v.dtype)}
  return list(found.values())
 persistent=unique_tensor_storages(list(net.state_dict().values())+list(vae.state_dict().values()))
 resident=unique_tensor_storages(list(net.parameters())+list(net.buffers())+list(vae.parameters())+list(vae.buffers()))
 result['state_storage']={'persistent_unique_bytes':sum(x['bytes'] for x in persistent),'persistent_unique_elements':sum(x['elements'] for x in persistent),'persistent_storages':persistent,'resident_including_nonpersistent_buffers_unique_bytes':sum(x['bytes'] for x in resident),'note':'nn.Parameter count excludes some codebook buffers; persistent storage includes them. Elements are mixed dtypes, not parameter count.'}
 result['runtime']={'python':sys.version,'platform':platform.platform(),'cuda_build':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'deterministic':torch.are_deterministic_algorithms_enabled(),'cpu_threads':torch.get_num_threads(),'CUBLAS_WORKSPACE_CONFIG':os.environ['CUBLAS_WORKSPACE_CONFIG']}
 stage=time.perf_counter()
 result['runtime']['pip_freeze']=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True,check=True).stdout.splitlines()
 costs['runtime_fingerprint_seconds']=time.perf_counter()-stage
 if not a.load_only:
  props=torch.cuda.get_device_properties(0)
  result['runtime']['gpu']={'name':props.name,'total_memory':props.total_memory,'capability':list(torch.cuda.get_device_capability(0))}
  stage=time.perf_counter()
  tokenizer=AutoTokenizer.from_pretrained(a.text_tokenizer,local_files_only=True,trust_remote_code=False)
  tokenizer.eos_token='<|endoftext|>';tokenizer.add_special_tokens({'pad_token':'<|pad|>'});assert tokenizer.pad_token_id==50257
  costs['text_tokenizer_construction_seconds']=time.perf_counter()-stage
  stage=time.perf_counter()
  vae=vae.to(a.device,dtype=torch.float32).eval();net=net.to(a.device,dtype=torch.float32).eval();allowance=allowance.to(a.device,dtype=torch.float64)
  def sync():
   if a.device.startswith('cuda'):torch.cuda.synchronize()
  sync();costs['gpu_transfer_seconds']=time.perf_counter()-stage
  costs['full_startup_seconds']=time.perf_counter()-PROCESS_START
  def generate(input_text=FIXTURE):
   # Count input tokenization, generation, final reranking, device synchronization and text decode.
   ids=tokenizer(input_text)['input_ids'];assert len(ids)<=512
   x=torch.tensor([ids+[tokenizer.pad_token_id]*(512-len(ids))],device=a.device)
   g=net.generate(None,x,temperature=1.,top_k=1,top_p=0.,cfg_scale=0.,cfg_null_eos=tokenizer.eos_token_id)
   _,zl=vae.vq.idx_to_zhats(g,return_zlevels=True);lg=vae.decoder(zl[-1]).double();B,T,V=lg.shape
   allow=(allowance[:V]*T)[None,:];cnt=torch.zeros(B,V,dtype=torch.float64,device=a.device);out=[]
   for t in range(T):
    step=lg[:,t,:].clone();step-=2.*(cnt-allow).clamp_min(0.);step+=1.*((T-t)/T)*(allow-cnt).clamp_min(0.)
    for j,h in enumerate(reversed(out[-4:])):step.scatter_add_(1,h[:,None],torch.full((B,1),-5.*.5**j,dtype=torch.float64,device=a.device))
    v=step.argmax(-1);out.append(v);cnt.scatter_add_(1,v[:,None],torch.ones(B,1,dtype=torch.float64,device=a.device))
   raw=torch.stack(out,1)[0].tolist();clean=raw[:raw.index(tokenizer.eos_token_id)] if tokenizer.eos_token_id in raw else raw
   text=tokenizer.decode([i for i in clean if i not in (tokenizer.eos_token_id,tokenizer.pad_token_id)],skip_special_tokens=True)
   return raw,text
  with torch.inference_mode():
   sync();start=time.perf_counter();raw,text=generate();sync();result['cold_generation_seconds']=time.perf_counter()-start
   times=[];same=True
   for _ in range(a.repeats):
    sync();start=time.perf_counter();r,t=generate();sync();times.append(time.perf_counter()-start);same &= r==raw
  # Frozen own DEV diagnostics. Exact-copy is supplied text, not learned capability.
  panel=DIAGNOSTIC_PANEL
  result['diagnostic_panel']=[]
  with torch.inference_mode():
   for row in panel:
    dtext=row['text'];ids=tokenizer(dtext)['input_ids'];assert len(ids)<=48
    x=torch.tensor([ids+[tokenizer.pad_token_id]*(48-len(ids))],device=a.device)
    sync();st=time.perf_counter();codes=vae.encode(x);rt_ids=vae.decode(codes).argmax(-1)[0].tolist();sync();rt_time=time.perf_counter()-st
    reconstruction=tokenizer.decode(rt_ids[:len(ids)],skip_special_tokens=True)
    sync();st=time.perf_counter();gids,gtext=generate(dtext);sync();gt=time.perf_counter()-st
    st=time.perf_counter();exact_copy=dtext[:];ct=time.perf_counter()-st
    result['diagnostic_panel'].append({'id':row['id'],'input':dtext,'supplied_fields':row['fields'],'input_token_ids':ids,'roundtrip_token_ids':rt_ids[:len(ids)],'vq_roundtrip_exact_tokens':rt_ids[:len(ids)]==ids,'vq_roundtrip_text':reconstruction,'vq_roundtrip_seconds_excluding_input_tokenization_tensor_creation_and_text_decode':rt_time,'generation_ids':gids,'generation':gtext,'generation_complete_call_seconds':gt,'generation_literal_field_retention':{key:value in gtext for key,value in row['fields'].items()},'exact_copy_control':exact_copy,'exact_copy_string_operation_seconds':ct,'control_status':'engineered supplied-text identity; does not learn/select/update facts'})
  result['diagnostic_scope']='Three freshly constructed synthetic records, not established unseen in external pretraining. Literal field omission is not automatically factual contradiction; no QA or CSM capability claim.'
  result['gpu_memory']={'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'current_allocated_bytes':torch.cuda.memory_allocated(),'current_reserved_bytes':torch.cuda.memory_reserved()}
  result.update(device=a.device,generated_ids=raw,summary=text,repeat_seconds=times,median_seconds=statistics.median(times),deterministic_repeat=same)
 result['cost_breakdown']=costs
 result['elapsed_before_result_serialization_seconds']=time.perf_counter()-PROCESS_START
 result['cost_exclusions']=['external download and Kaggle queue/upload time','interpreter startup before first perf_counter','final JSON serialization/write and process teardown'];result['execution_counts']={'benchmark_generation_calls':0 if a.load_only else 4,'additional_diagnostic_generation_calls':0 if a.load_only else 3,'vq_roundtrip_calls':0 if a.load_only else 3}
 pathlib.Path(a.out).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
