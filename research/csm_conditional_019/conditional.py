"""Supplied conditional semantics. No learned comparison claim or Lean claim."""
import torch
import executor
def apply(state,op,dst,src,value,expected,learned_gate=None):
 # All operands validated even for a false branch: malformed programs reject.
 if expected.dtype!=torch.long or expected.shape!=op.shape or not bool(((expected>=1)&(expected<=4)).all()):raise ValueError('expected payload')
 if not bool(((op>=0)&(op<=6)).all()):raise ValueError('opcode')
 mask=op==6
 if not bool((~mask|((src>=0)&(src<state[0].shape[1]))).all()):raise ValueError('conditional source')
 if not bool((~mask|((value>=1)&(value<=4))).all()):raise ValueError('conditional new payload')
 if learned_gate is None:
  gate=(executor.read(state,torch.where(mask,src,torch.zeros_like(src)))==expected) if bool(mask.any()) else torch.zeros_like(mask)
 else:
  if learned_gate.dtype!=torch.bool or learned_gate.shape!=op.shape:raise ValueError('gate')
  gate=learned_gate
 actual=torch.where(op==6,torch.where(gate,torch.ones_like(op),torch.zeros_like(op)),op)
 return executor.step(state,actual,dst,src,value)
