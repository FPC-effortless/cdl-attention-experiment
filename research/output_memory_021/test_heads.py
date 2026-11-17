import unittest
from pathlib import Path
import torch
from heads import *
from conditioner import *

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'goal-campaign/restored/LANGUAGE_BASELINE_001/replay_sources'

class Tests(unittest.TestCase):
 def setUp(self):
  torch.set_num_threads(1);torch.manual_seed(7)
  self.head=OutputHead(12,width=16,layers=2,max_length=16).eval();self.c=torch.randn(2,12)
 def test_cached_ar_logits(self):
  x=torch.randint(0,259,(2,12));full=self.head(x,self.c,causal=True);cache=None;out=[]
  for i in range(12):
   y,cache=self.head.ar_step(x[:,i:i+1],self.c,cache);out.append(y)
  self.assertTrue(torch.allclose(full,torch.cat(out,1),atol=1e-6,rtol=1e-5))
 def test_causal_no_future_target(self):
  x=torch.randint(0,259,(2,12));y=x.clone();y[:,6:]=17
  self.assertTrue(torch.equal(self.head(x,self.c,causal=True)[:,:6],self.head(y,self.c,causal=True)[:,:6]))
 def test_rng_reorder_diffusion(self):
  a,_=generate_diffusion(self.head,self.c,8,4,['a','b'],41)
  b,_=generate_diffusion(self.head,self.c.flip(0),8,4,['b','a'],41)
  self.assertTrue(torch.equal(a,b.flip(0)));self.assertFalse((a==MASK).any())
 def test_rng_reorder_ar(self):
  a,_=generate_ar(self.head,self.c,8,['a','b'],41)
  b,_=generate_ar(self.head,self.c.flip(0),8,['b','a'],41)
  self.assertTrue(torch.equal(a,b.flip(0)))
 def test_padding_insert(self):
  for mode in ['ar','diff']:
   if mode=='ar':
    a,_=generate_ar(self.head,self.c[:1],8,['a'],41)
    b,_=generate_ar(self.head,self.c.flip(0),8,['pad','a'],41,torch.tensor([False,True]))
   else:
    a,_=generate_diffusion(self.head,self.c[:1],8,4,['a'],41)
    b,_=generate_diffusion(self.head,self.c.flip(0),8,4,['pad','a'],41,torch.tensor([False,True]))
   self.assertTrue(torch.equal(a[0],b[1]));self.assertTrue((b[0]==PAD).all())
 def test_no_mutation(self):
  old=self.c.clone();generate_diffusion(self.head,self.c,8,4,['a','b']);generate_ar(self.head,self.c,8,['a','b'])
  self.assertTrue(torch.equal(old,self.c))
 def test_training_gradient(self):
  self.head.train();clean=torch.randint(0,258,(2,8));loss=masked_loss(self.head,clean,self.c,.5,torch.Generator().manual_seed(1));loss.backward()
  self.assertTrue(torch.isfinite(loss));self.assertGreater(self.head.output.weight.grad.abs().sum(),0)
 def test_invalid_arguments(self):
  for t in [float('nan'),float('inf'),-1]:
   with self.assertRaises(ValueError):self.head(torch.ones(2,8,dtype=torch.long),self.c,t)
  with self.assertRaises(ValueError):generate_ar(self.head,self.c,17,['a','b'])
  with self.assertRaises(ValueError):generate_ar(self.head,self.c,8,['a','b'],active=torch.ones(2))
  with self.assertRaises(ValueError):generate_diffusion(self.head,self.c,8,0,['a','b'])
 def test_nonfinite_logits(self):
  self.head.output.bias.data[0]=float('nan')
  with self.assertRaises(ValueError):generate_diffusion(self.head,self.c,8,2,['a','b'])
 def test_decode(self):
  self.assertEqual(decode_task(torch.tensor([65,EOS,PAD])),(b'A','terminated'))
  self.assertEqual(decode_task(torch.tensor([65,PAD]))[1],'malformed')
  self.assertEqual(decode_task(torch.tensor([65]))[1],'unterminated')
 def test_real_conditioners_and_batch_padding(self):
  for seed in (11,23,37):
   folder=BASE/f'csm-{seed}';m,receipt=load_conditioner(folder/'checkpoint.pt',folder/'native_csm.py')
   a=prefix_condition(m,[b'Code A.']);b=prefix_condition(m,[b'x',b'Code A.',b'longer history with distractor'])
   self.assertEqual(receipt['condition_dim'],392)
   self.assertTrue(torch.allclose(a[0],b[1],atol=1e-6,rtol=1e-5))
   self.assertEqual(receipt['seed'],seed)
 def test_session_consumption_matches_prefix(self):
  folder=BASE/'csm-11';m,_=load_conditioner(folder/'checkpoint.pt',folder/'native_csm.py')
  session=FrozenSession(m);session.consume(b'Code A.');session.consume(b' Answer:Z')
  expected=prefix_condition(m,[b'Code A. Answer:Z'])
  self.assertTrue(torch.equal(expected,session.feature()));self.assertEqual(session.ingested_bytes,16)
  feature=session.feature();feature.zero_();self.assertTrue(torch.equal(expected,session.feature()))
 def test_sentence_capacity64_and_condition_padding(self):
  head=OutputHead(704,width=16,layers=2,max_length=64).eval();condition=torch.zeros(2,704);condition[:,:392]=torch.randn(2,392)
  x=torch.randint(0,259,(2,64));full=head(x,condition,causal=True);cache=None;out=[]
  for i in range(64):
   z,cache=head.ar_step(x[:,i:i+1],condition,cache);out.append(z)
  self.assertTrue(torch.allclose(full,torch.cat(out,1),atol=1e-6,rtol=1e-5))
  raw,r=generate_diffusion(head,condition,64,16,['a','b'],91)
  self.assertEqual(tuple(raw.shape),(2,64));self.assertFalse((raw==MASK).any());self.assertEqual(r['position_evaluations'],2048)
  a,ar=generate_ar(head,condition,64,['a','b'],91)
  self.assertEqual(tuple(a.shape),(2,64));self.assertEqual(ar['position_evaluations'],2*ar['head_calls']);self.assertLessEqual(ar['head_calls'],64)
 def test_ar_early_eos_and_inactive_cost(self):
  self.head.output.weight.data.zero_();self.head.output.bias.data.fill_(-100)
  self.head.output.bias.data[EOS]=100
  a,r=generate_ar(self.head,self.c,16,['a','b'],31,torch.tensor([True,False]),greedy=True)
  self.assertEqual(r['head_calls'],1);self.assertEqual(r['active_position_evaluations'],1)
  self.assertEqual(r['padded_position_evaluations'],1);self.assertEqual(a[0,0],EOS)
  self.assertTrue((a[:,1:]==PAD).all());self.assertTrue((a[1]==PAD).all())
  self.assertEqual(decode_task(a[0]),(b'','terminated'))
 def test_ar_malformed_pad_terminates(self):
  self.head.output.weight.data.zero_();self.head.output.bias.data.fill_(-100)
  self.head.output.bias.data[PAD]=100
  a,r=generate_ar(self.head,self.c,16,['a','b'],greedy=True)
  self.assertEqual(r['head_calls'],1);self.assertEqual(decode_task(a[0])[1],'malformed')
 def test_ar_no_active_no_work(self):
  a,r=generate_ar(self.head,self.c,16,['a','b'],active=torch.zeros(2,dtype=torch.bool))
  self.assertEqual(r['head_calls'],0);self.assertTrue((a==PAD).all())
 def test_prefix_only_adapter_signature(self):
  import inspect
  self.assertEqual(list(inspect.signature(prefix_condition).parameters),['model','prefixes'])

if __name__=='__main__':unittest.main()
