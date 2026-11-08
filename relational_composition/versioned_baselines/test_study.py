import random
import unittest
import torch
from .study import ARMS, Config, NAMES, PLACES, VOCAB, build, dataset, make_group, objective, take

class IntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_counterfactual_twins_and_snapshot_oracle(self):
        for heldout in (False, True):
            for seed in range(20):
                group = make_group(random.Random(seed), heldout)
                for item in group:
                    first, second = item['support']
                    r = item['records']
                    self.assertEqual(r[first][4], VOCAB[NAMES[item['entity']]])
                    self.assertEqual(r[first][7], VOCAB[NAMES[item['person']]])
                    self.assertEqual(r[second][4], VOCAB[NAMES[item['person']]])
                    self.assertEqual(r[second][7], VOCAB[PLACES[item['answer']]])
                    self.assertEqual(r[first][2], item['query'][2])
                    self.assertEqual(r[second][2], item['query'][2])
                    self.assertEqual(item['person'] == (item['entity']+1)%4, heldout)
                    self.assertEqual(item['query'], group[0]['query'])
                    diffs = [(i,j) for i in range(16) for j in range(12) if r[i][j] != group[0]['records'][i][j]]
                    self.assertEqual(diffs, [] if item['answer']==0 else [(second,7)])

    def test_balanced_memory_off_bound(self):
        data = dataset(99, 10)
        model = build('query_only', Config()).eval()
        logits, _ = model(data['records'], data['query'])
        self.assertEqual(float(logits.argmax(-1).eq(data['answer']).float().mean()), 0.25)

    def test_forward_has_no_label_inputs_and_auxiliary_gradients(self):
        data = dataset(88, 2)
        for arm in ARMS:
            model = build(arm, Config())
            a, reads = model(data['records'], data['query'])
            self.assertEqual(tuple(a.shape), (8,4))
            if reads is not None:
                self.assertEqual(tuple(reads.shape), (8,2,16))
                torch.testing.assert_close(reads.sum(-1), torch.ones(8,2))
            loss, _, _ = objective(model, data, 1.0 if arm.endswith('_support') else 0.0)
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            # Changing support metadata cannot alter predictions.
            data['support'] = data['support'].flip(-1)
            b, _ = model(data['records'], data['query'])
            torch.testing.assert_close(a,b, rtol=0, atol=0)

    def test_dataset_deterministic(self):
        a, b = dataset(10, 2), dataset(10, 2)
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)

if __name__ == '__main__':
    unittest.main()
