import unittest
import torch
from ..versioned_baselines.study import dataset
from .model import SentenceLRC, support_loss, gold_route_overrides
from .diagnostics import credit_diagnostics

class LRCIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_oracle_mapping_retrieves_both_supports_through_index(self):
        for seed in range(3):
            torch.manual_seed(seed)
            model=SentenceLRC()
            data=dataset(100+seed,2)
            base=model(data['records'],data['query'])
            forced=model(data['records'],data['query'],overrides=gold_route_overrides(base,data['support']))
            for refs in forced.operands:
                torch.testing.assert_close(refs[:,-1],data['support'],rtol=0,atol=0)
                self.assertTrue((refs<torch.arange(17)[None,:,None]).all())
            self.assertLessEqual(forced.metrics['candidates'],2*forced.metrics['reads'])

    def test_supervision_has_writer_gradients_without_extra_labels(self):
        torch.manual_seed(91)
        model=SentenceLRC()
        data=dataset(101,2)
        out=model(data['records'],data['query'])
        loss=support_loss(out,data['support'])
        loss.backward()
        for layer in model.layers:
            self.assertGreater(float(layer.write_address.weight.grad.abs().sum()),0)
            self.assertGreater(float(layer.write_relation.weight.grad.abs().sum()),0)
        again=model(data['records'],data['query'])
        torch.testing.assert_close(out.logits,again.logits,rtol=0,atol=0)

    def test_disabled_memory_is_exact_chance_on_twins(self):
        data=dataset(103,4)
        out=SentenceLRC()(data['records'],data['query'],memory='disabled')
        self.assertEqual(float(out.logits[:,-1].argmax(-1).eq(data['answer']).float().mean()),.25)
        self.assertTrue(all(x.eq(-1).all() for x in out.operands))

    def test_controlled_credit_conjunction_is_nonvacuous(self):
        torch.manual_seed(65)
        result=credit_diagnostics(SentenceLRC(),dataset(204,4),seed=71,examples=16,repeats=1)
        for item in result['controlled_conjunction']:
            self.assertGreater(item['eligible_examples'],0)
            self.assertGreater(item['joint_nonzero'],0)
            if item['required_repairs']>1:
                self.assertEqual(item['single_nonzero'],0)
            else:
                self.assertEqual(item['single_nonzero'],item['joint_nonzero'])

if __name__=='__main__':
    unittest.main()
