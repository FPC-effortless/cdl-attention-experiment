import unittest
import torch
from model import SlotStore
class ModelTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(123);torch.set_num_threads(1);self.model=SlotStore().eval()
    def test_parameters(self): self.assertEqual(sum(p.numel() for p in self.model.parameters()),18254)
    def test_causality_and_step_parity(self):
        ids=torch.randint(19,(2,12));a,_=self.model.sequence(ids,all_outputs=True)
        b,_=self.model.sequence(ids[:,:7],all_outputs=True)
        self.assertTrue(torch.equal(a[:,:7],b))
        state=self.model.init_state(2,'cpu');values=[]
        for token in ids.unbind(1): x,state=self.model.step(token,state);values.append(x)
        self.assertTrue(torch.equal(a,torch.stack(values,1)))
    def test_no_admission_preserves_values(self):
        with torch.no_grad():self.model.value_edit.weight.zero_();self.model.value_edit.bias.copy_(torch.tensor([100.,-100.]))
        state=self.model.init_state(2,'cpu');state[0].normal_();old=state[0].clone()
        _,state=self.model.step(torch.tensor([4,5]),state)
        self.assertTrue(torch.equal(old,state[0]))
    def test_labels_not_accepted(self):
        with self.assertRaises(TypeError):self.model.sequence(torch.tensor([[1,2,3]]),targets=torch.tensor([1]))
    def test_fresh_states_do_not_alias(self):
        a=self.model.init_state(1,'cpu');b=self.model.init_state(1,'cpu');a[0].fill_(2)
        self.assertEqual(float(b[0].sum()),0)
if __name__=='__main__':unittest.main()
