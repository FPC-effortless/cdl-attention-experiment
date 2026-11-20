"""Adversarial pair-roster tests; no model fits or development outcomes."""
import copy,unittest
from unittest.mock import patch
from test_adapters import g,Head,wrong_generate
from evaluate_sessions import evaluate_pair
from score_sessions import score
class GlobalPairTests(unittest.TestCase):
    def setUp(self):
        self.a=g.episode_pair('software',0,0);self.b=g.episode_pair('software',1,0)
        with patch('evaluate_sessions.generate_ar',wrong_generate):
            self.ra=evaluate_pair(None,Head(),self.a,condition='exact')
            self.rb=evaluate_pair(None,Head(),self.b,condition='exact')
    def test_valid_multiple_groups(self):
        summary=score(self.a+self.b,[self.ra,self.rb])
        self.assertEqual(summary['counterfactual_first_snapshot_pairs'],2)
    def test_distinct_ids_repeated_pair_across_batches(self):
        # Four unique session IDs, but only one alleged independent pair.
        for row in self.b:row['pair_id']=self.a[0]['pair_id']
        for row in self.rb['rows']:row['pair_id']=self.a[0]['pair_id']
        with self.assertRaisesRegex(ValueError,'global roster'):score(self.a+self.b,[self.ra,self.rb])
    def test_duplicate_variant_in_roster(self):
        self.a[1]['variant']=0;self.ra['rows'][1]['variant']=0
        with self.assertRaisesRegex(ValueError,'global roster'):score(self.a,[self.ra])
    def test_duplicate_variant_in_results(self):
        self.ra['rows'][1]['variant']=0
        with self.assertRaisesRegex(ValueError,'variant mismatch'):score(self.a,[self.ra])
    def test_result_pair_substitution(self):
        for row in self.rb['rows']:row['pair_id']=self.a[0]['pair_id']
        with self.assertRaisesRegex(ValueError,'pair ID'):score(self.a+self.b,[self.ra,self.rb])
    def test_split_pairs_rejected(self):
        x=copy.deepcopy(self.ra);y=copy.deepcopy(self.rb)
        x['rows']=[self.ra['rows'][0],self.rb['rows'][0]]
        y['rows']=[self.ra['rows'][1],self.rb['rows'][1]]
        with self.assertRaisesRegex(ValueError,'split/incomplete'):score(self.a+self.b,[x,y])
    def test_valid_128_pair_count(self):
        from data_adapter import render_target
        records=[];results=[]
        for i in range(128):
            pair=g.episode_pair('software128',i,0);records.extend(pair)
            result={'kind':'ar','steps':None,'condition':'exact','intervention':'full','output_capacity':64,'seconds':0,'rows':[]}
            for r in pair:
                answers=[]
                for q,a in enumerate(r['answers']):
                    text=render_target(a);data=list(text.encode());raw=data+[256]+[257]*(63-len(data))
                    answers.append({'query_index':q,'identity':a['identity'],'payload_target':a['target'],'raw_tokens':raw,'correct':True,'payload_correct':True,'target':text,'prediction_bytes':data,'prediction':text,'payload_prediction':a['target']})
                result['rows'].append({'id':r['id'],'pair_id':r['pair_id'],'variant':r['variant'],'answers':answers})
            results.append(result)
        summary=score(records,results)
        self.assertEqual(summary['counterfactual_first_snapshot_pairs'],128)
        self.assertEqual(summary['sessions'],256);self.assertTrue(summary['gate_pass'])
    def test_scientific_evaluator_rejects_bad_roster(self):
        from evaluate_scientific import evaluate_batch
        records=copy.deepcopy(self.a+self.b)
        for r in records[2:]:r['pair_id']=records[0]['pair_id']
        with self.assertRaisesRegex(ValueError,'exactly variants'):evaluate_batch(None,None,records)
        records=copy.deepcopy(self.a);records[1]['variant']=0
        with self.assertRaisesRegex(ValueError,'exactly variants'):evaluate_batch(None,None,records)
        records=copy.deepcopy(self.a);records[1]['id']=records[0]['id']
        with self.assertRaisesRegex(ValueError,'duplicate session'):evaluate_batch(None,None,records)
if __name__=='__main__':unittest.main()
