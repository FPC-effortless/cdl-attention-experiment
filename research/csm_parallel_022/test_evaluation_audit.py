"""Independent adversarial scoring audit; oracle rows are software fixtures only."""
import copy,unittest
from data import episode_pair,regions_pair,OPS,external_events,payload_text
from evaluate import commands,score
from executor import reference

def oracle_rows(records):
    rows=[]
    for record in records:
        actions=[{'op':OPS[e['labels']['op']],'valid':True,'spans':{k:(None if v==[-1,-1] else list(v)) for k,v in e['labels']['spans'].items()}} for e in record['events']]
        ops,n,banks,errors,registry=commands(external_events(record),actions)
        result=reference(ops,n);state=None;guards=[]
        for op in ops:
            guards.append(reference([[5,op[2],0,0,0]],n,state)['answers'][0]==op[4] if op[0]==6 else None)
            state=reference([op],n,state)['state']
        rendered=[None if a==-2 else 'No code is currently assigned.' if a is None else 'The current code is '+payload_text(a)+'.' for a in result['answers']]
        rows.append({'id':record['id'],'pair_id':record['pair_id'],'variant':record['variant'],'actions':actions,'typed_commands':ops,'registry':registry,'errors':errors,'answers':result['answers'],'rendered_answers':rendered,'predicted_guards':guards,'final_state':[list(x) for x in result['state']]})
    return rows
class EvaluationAuditTests(unittest.TestCase):
    def setUp(self):self.records=episode_pair('audit',44);self.rows=oracle_rows(self.records)
    def test_six_query_metrics(self):
        result=score(self.records,self.rows)
        self.assertEqual(result['queries'],12);self.assertEqual(result['pairs'],1)
        self.assertEqual(result['absence_queries'],2);self.assertEqual(result['nonabsence_queries'],10)
        self.assertEqual(result['conditional_decisions'],2);self.assertTrue(result['qualified'])
    def test_independent_regions_24_queries_not_4_subjects(self):
        records=regions_pair('audit_regions',4);result=score(records,oracle_rows(records))
        self.assertEqual(result['queries'],48);self.assertEqual(result['pairs'],1)
        self.assertEqual(result['absence_queries'],8);self.assertEqual(result['nonabsence_queries'],40)
        self.assertEqual(result['conditional_decisions'],8);self.assertTrue(result['qualified'])
    def test_global_reused_pair_id(self):
        records=self.records+episode_pair('audit',45)
        for r in records[2:]:r['pair_id']=records[0]['pair_id']
        with self.assertRaises(ValueError):score(records,oracle_rows(records))
    def test_variant_substitution(self):
        self.rows[1]['variant']=0
        with self.assertRaises(ValueError):score(self.records,self.rows)
    def test_duplicate_prediction_ids(self):
        self.rows[1]['id']=self.rows[0]['id']
        with self.assertRaises(ValueError):score(self.records,self.rows)
    def test_forged_commands_rejected(self):
        self.rows[0]['typed_commands'][0]=[0,0,0,0,0]
        with self.assertRaises((ValueError,RuntimeError)):score(self.records,self.rows)
    def test_forged_answers_rejected_not_merely_rescored(self):
        index=next(i for i,e in enumerate(self.records[0]['events']) if e['gold']['op_name']=='QUERY')
        self.rows[0]['answers'][index]=None
        with self.assertRaises((ValueError,RuntimeError)):score(self.records,self.rows)
    def test_forged_rendered_answer_rejected(self):
        index=next(i for i,e in enumerate(self.records[0]['events']) if e['gold']['op_name']=='QUERY')
        self.rows[0]['rendered_answers'][index]='The current code is WRONG.'
        with self.assertRaises((ValueError,RuntimeError)):score(self.records,self.rows)
    def test_forged_guard_rejected(self):
        index=next(i for i,e in enumerate(self.records[0]['events']) if e['gold']['op_name']=='IFSET')
        self.rows[0]['predicted_guards'][index]=not self.rows[0]['predicted_guards'][index]
        with self.assertRaises((ValueError,RuntimeError)):score(self.records,self.rows)
    def test_forged_final_state_rejected(self):
        self.rows[0]['final_state'][0][0]=0
        with self.assertRaises((ValueError,RuntimeError)):score(self.records,self.rows)
if __name__=='__main__':unittest.main()
