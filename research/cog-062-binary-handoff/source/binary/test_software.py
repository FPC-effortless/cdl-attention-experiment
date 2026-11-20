import json,os,tempfile,unittest
from pathlib import Path
import numpy as np
from model import WIDTH, hit_supervision, meta_features, meta_text, ordered_text, rank_targets, shortlist
from score import main
from fit import validate_fit_paths

class SoftwareOnly(unittest.TestCase):
    def test_exact_evaluation_path_check(self):
        validate_fit_paths(['RUNTIME.json','data/A.json','data/B.json'])
        for name in ('E.json','E-input.json','E-gold.json','E-overlap.json'):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'sealed out'):
                validate_fit_paths(['RUNTIME.json','data/'+name])

    def test_candidate_hit_supervision_preserves_order(self):
        mask, labels=hit_supervision(np.array([False,True,True,False,True]),np.array([2,1,0]))
        self.assertEqual(mask.tolist(),[False,False,True,False,True])
        self.assertEqual(labels.tolist(),[1,0])

    def test_same_strings_and_fixed_width(self):
        r={'text':['hotel please','north please','do you mean west']}
        self.assertNotEqual(ordered_text(r,'ordered'),ordered_text(r,'canonical'))
        self.assertEqual(sorted(ordered_text(r,'ordered')),sorted(ordered_text(r,'canonical')))
        c,p=shortlist(np.array([[.1,.2,.7]],dtype=float))
        for m in ('canonical','ordered'):
            x=meta_features(meta_text([r],m),c,p)
            self.assertEqual(x.shape,(1,WIDTH))
        self.assertEqual(rank_targets(np.array(['north','west','east']),np.array([['north','west']]*3)).tolist(),[0,1,2])

    def test_scoring_denominators_only_constructed_fixture(self):
        with tempfile.TemporaryDirectory() as t:
            d=Path(t);os.environ['TRAINLAB_OUTPUT_DIR']=t
            i={'dialogue':'fake','group':'fake','turn':2}
            p={'top2':['<hold>','west'],'canonical':'<hold>','ordered':'west','order_removed':'<hold>'}
            rows=[[i|{'slots':[p,p]}],[i|{'delta':['east','<hold>'],'prior':['<empty>','west']}],[i|{'exact_tuple_overlap':True,'group_contains_tuple_overlap':True}]]
            paths=[d/f'x{j}.json' for j in range(3)]
            for x,z in zip(paths,rows):x.write_text(json.dumps(z))
            main(*paths)
            out=json.loads((d/'scores.json').read_bytes())
            self.assertEqual(out['changed_shortlist_miss'],1)
            self.assertEqual(out['cohorts']['changed']['rows'],1)
            self.assertEqual(out['cohorts']['candidate:miss']['rows'],1)
            self.assertEqual(out['cohorts']['candidate:hit']['rows'],0)
            self.assertEqual(out['flip_tradeoff']['ordered']['base_wrong_now_right'],0)
            self.assertEqual(out['total_rank2_switch_accounting']['ordered']['both_miss_switch'],1)
            self.assertEqual(out['total_rank2_switch_accounting']['ordered']['unchanged_false_update'],1)
            self.assertEqual(out['cohorts']['overlap:tuple']['rows'],1)
            self.assertEqual(out['cohorts']['changed']['support'],'INSUFFICIENT_SUPPORT')
            self.assertFalse(out['directional_gate_met'])
            self.assertEqual(out['unchanged_false_update_proposals']['price']['ordered'],1)
