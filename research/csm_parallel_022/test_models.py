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
