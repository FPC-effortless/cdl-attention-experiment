import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from cache_scientific import cache
from evaluate_scientific import evaluate_pair
from resolved_state import ADAPTER_VERSION
from test_adapters import FakeBatch,Head,wrong_generate,g
class ScientificAdapterTests(unittest.TestCase):
    def test_versioned_cache_resolves_links(self):
        records=g.episode_pair('train',9,0)
        with tempfile.TemporaryDirectory() as d,patch('cache_scientific.BatchedFrozenSession',FakeBatch):
            path=Path(d)/'cache.pt';cache(None,records,path)
            saved=torch.load(path,weights_only=True)
            self.assertEqual(saved['bindings']['exact_adapter_version'],ADAPTER_VERSION)
            self.assertEqual(saved['exact_resolution']['feature_calls'],12)
            self.assertEqual(saved['exact_resolution']['resolved_records'],192)
            # First query features expose both the snapshot and resolved live-reference payloads.
            first=saved['exact_features'][0];flags=first[:352].reshape(16,22)
            refs=flags[:,2]==1
            self.assertTrue((flags[refs,3:6]>0).all())
    def test_scientific_eval_cost_version(self):
        records=g.episode_pair('dev_id',9,0)
        with patch('evaluate_scientific.generate_ar',wrong_generate):result=evaluate_pair(None,Head(),records,condition='exact')
        self.assertEqual(result['exact_adapter_version'],ADAPTER_VERSION)
        self.assertEqual(sum(r['resolved_records'] for r in result['exact_resolution']),192)
        self.assertTrue(all(r['feature_seconds']>=0 for r in result['exact_resolution']))
        self.assertFalse(any(r['session_correct'] for r in result['rows']))
if __name__=='__main__':unittest.main()
