import unittest
import torch
from models import *

class Tests(unittest.TestCase):
 def setUp(self):torch.set_num_threads(1);torch.manual_seed(22)
 def test_chunk_parity_both_architectures(self):
  texts=['Set east/Mira to 00A.','Copy east/Mira into west/Lena.','What is west/Lena?','If A is 0aX, set B to 9zT.']*9
  for arm in ('parallel','gru'):
   model=SentenceParser(Config(arm=arm,max_length=64)).eval()
   ref,act=infer_chunks(model,texts,1)
   for n in (8,32,128):
    z,b=infer_chunks(model,texts,n);self.assertTrue(parity_guard(ref,act,z,b)['same_actions'])
 def test_failclosed_badspan(self):
  model=SentenceParser(Config(max_length=64)).eval();x,n=tensorize_texts(['Set A to B.'],max_length=64);z=model(x,n)
  z['op'].fill_(-100);z['op'][0,1]=100
  z['dst_start'].fill_(-100);z['dst_end'].fill_(-100);z['dst_start'][0,8]=100;z['dst_end'][0,2]=100
  self.assertFalse(decode_logits(z,n)[0]['valid'])
 def test_argmax_guard_rejects(self):
  model=SentenceParser(Config(max_length=64)).eval();a,b=infer_chunks(model,['Set A to B.'],1)
  with self.assertRaises(RuntimeError):parity_guard(a,b,a,[{'op':'NOOP'}])
 def test_nonfinite_guard_rejects(self):
  model=SentenceParser(Config(max_length=64)).eval();a,b=infer_chunks(model,['Set A to B.'],1)
  a['dst_start'][0,0]=float('nan')
  with self.assertRaises(RuntimeError):parity_guard(a,b,a,b)
  with self.assertRaises(ValueError):decode_logits(a,torch.tensor([11]))
 def test_global_context_changes_relative_roles_and_gradient(self):
  m=SentenceParser(Config(arm='gru',max_length=64)).eval();w=m.config.width
  scorer=m.role['dst_start'];scorer.weight.data.zero_();scorer.bias.data.zero_();scorer.weight.data[0,w]=1
  local=torch.zeros(1,2,w);local[0,0,0]=1;local[0,1,0]=2
  global_context=torch.ones(1,w,requires_grad=True);mask=torch.ones(1,2,dtype=torch.bool)
  a=m.role_logits(local,global_context,mask)['dst_start']
  b=m.role_logits(local,global_context*2,mask)['dst_start']
  self.assertNotEqual(float((a[0,1]-a[0,0]).detach()),float((b[0,1]-b[0,0]).detach()))
  F.cross_entropy(a,torch.tensor([0])).backward()
  self.assertGreater(abs(float(global_context.grad[0,0])),.1)
 def test_gru_late_text_changes_early_relative_scores(self):
  m=SentenceParser(Config(arm='gru',max_length=64)).eval()
  x,n=tensorize_texts(['Copy A from B.','Copy A into B.'],max_length=64)
  z=m(x,n)['dst_start']
  # Earlier byte representations match up to the changed suffix, but role
  # scores can use the later sentence information through global interaction.
  self.assertGreater(abs(float((z[0,5]-z[0,6])-(z[1,5]-z[1,6]))),1e-6)
 def test_boundaries_no_truncation(self):
  for text in ('','a'*65):
   with self.assertRaises(ValueError):tensorize_texts([text],max_length=64)
 def test_supervision_gradient(self):
  for arm in ('parallel','gru'):
   model=SentenceParser(Config(arm=arm,max_length=64));x,n=tensorize_texts(['Set A to B.'],max_length=64)
   labels=[{'op':1,'spans':{'dst':[4,5],'value':[9,10]}}]
   loss=supervised_loss(model(x,n),labels);loss.backward();self.assertTrue(torch.isfinite(loss))
 def test_padding_not_operand(self):
  m=SentenceParser(Config(max_length=64)).eval();x,n=tensorize_texts(['A'],max_length=64);z=m(x,n)
  self.assertTrue(torch.isneginf(z['dst_start'][0,1:]).all())
 def test_future_sentence_cannot_change_parse(self):
  m=SentenceParser(Config(max_length=64)).eval()
  a,b=infer_chunks(m,['Set A to B.'],1);c,d=infer_chunks(m,['Set A to B.','Delete A.'],8)
  parity_guard(a,b,{k:v[:1] for k,v in c.items()},d[:1])

if __name__=='__main__':unittest.main()
