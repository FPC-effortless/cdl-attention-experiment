"""Explicit, scoped edits of the archived CSM for mechanistic diagnostics."""
import sys,types,hashlib
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'csm_parity_002'))
from regulated import Regulated,base,source
ARMS=('hard','soft_routes','soft_admission','residual_front')
MODULES={}
def replace_once(text,old,new):
 assert text.count(old)==1,old
 return text.replace(old,new)
def make(arm):
 assert arm in ARMS
 m=Regulated(base.Config(vocab_size=256),'multi').float()
 if arm=='hard':return m
 if arm not in MODULES:
  text=source.decode()
  if arm=='soft_routes':
   text=replace_once(text,'choose = lambda logits: hard_choice(logits, cfg.temperature)','choose = lambda logits: (logits / cfg.temperature).softmax(-1)')
   for head in ('value_edit','link_edit'):
    text=replace_once(text,f'choose(self.{head}(control))',f'hard_choice(self.{head}(control), cfg.temperature)')
  if arm=='soft_admission':
   text=replace_once(text,'choose(self.value_edit(control))','(self.value_edit(control) / cfg.temperature).softmax(-1)')
  if arm=='residual_front':
   text=replace_once(text,'logits = self.output(updated)','logits = self.output((updated + front) / (2 ** 0.5))')
  mod=types.ModuleType('_csm_'+arm);sys.modules[mod.__name__]=mod;exec(compile(text,mod.__name__,'exec'),mod.__dict__);MODULES[arm]=mod
  mod.source_sha256=hashlib.sha256(text.encode()).hexdigest()
 original=m.core.state_dict();m.core=MODULES[arm].RecordRewrite(base.Config(vocab_size=256));m.core.load_state_dict(original)
 return m
