import copy,json,tempfile,unittest
from pathlib import Path
from data import *
class DataTests(unittest.TestCase):
    def test_spans_and_reference_strings(self):
        for wording in ['train','dev']:
            for comp in [False,True]:
                for banks in [False,True]:
                    rows=episode_pair('software',17,wording,'dev',comp,2,banks)
                    validate_roster(rows)
                    for row in rows:
                        state={}
                        def read(name):
                            seen=set()
                            while name in state:
                                if name in seen:return None
                                seen.add(name);kind,value=state[name]
                                if kind=='value':return value
                                name=value
                            return None
                        for event in row['events']:
                            text=event['text'];labels=event['labels'];gold=event['gold'];args={}
                            for field,span in labels['spans'].items():
                                args[field]=None if span==[-1,-1] else text[slice(*span)]
                                target=gold['dst_name'] if field=='dst' else gold['src_name'] if field=='src' else gold['payload'] if field=='value' else gold['expected']
                                self.assertEqual(args[field],target)
                                tok=labels['token_spans'][field]
                                if tok!=[-1,-1]:self.assertEqual(event['tokenization']['tokens'][tok[0]],target)
                            op=OPS[labels['op']];d,s,v,c=(args[f] for f in FIELDS)
                            if op=='SET':state[d]=('value',v)
                            elif op=='COPY':
                                value=read(s)
                                if value is None:state.pop(d,None)
                                else:state[d]=('value',value)
                            elif op=='LINK':state[d]=('ref',s)
                            elif op=='DELETE':state.pop(d,None)
                            elif op=='QUERY':self.assertEqual(read(d),gold['answer'])
                            elif op=='IFSET':
                                guard=read(s)==c;self.assertEqual(guard,gold['guard'])
                                if guard:state[d]=('value',v)
    def test_eval_accessor_has_no_labels(self):
        r=episode_pair('software',1)[0];external=external_events(r)
        modified=copy.deepcopy(r)
        for e in modified['events']:e['labels']={'invalid':True};e['gold']={'answer':'LEAK'}
        self.assertEqual(external,external_events(modified));self.assertTrue(all(type(x)is str for x in external))
    def test_global_pair_mutations(self):
        a=episode_pair('software',1);b=episode_pair('software',2)
        for r in b:r['pair_id']=a[0]['pair_id']
        with self.assertRaises(ValueError):validate_roster(a+b)
        a=episode_pair('software',1);a[1]['variant']=0
        with self.assertRaises(ValueError):validate_roster(a)
    def test_payload_bytes_preserved(self):
        for value in ['000','00A','A01','ZZZ']:self.assertEqual(payload_text(payload_int(value)),value)
    def test_cross_bank_has_same_basename(self):
        r=episode_pair('software',5,cross_bank=True)[0]
        names={e['gold']['dst_name'] for e in r['events'] if e['gold']['dst_name']}
        self.assertEqual(len(names),4);self.assertEqual(len({n.split('/')[1] for n in names}),2)
    def test_manifest_reproducible_no_overlap(self):
        with tempfile.TemporaryDirectory() as d:
            a=build(Path(d)/'a',4,2);b=build(Path(d)/'b',4,2)
            self.assertEqual(a,b);self.assertEqual(sum(p['sessions'] for p in a['panels']),36)
    def test_interleaved_independent_regions(self):
        rows=regions_pair('software_regions',3);counts=validate_roster(rows)
        self.assertEqual(counts['queries'],48);self.assertEqual(counts['independent_pairs'],1)
        for row in rows:
            banks=set()
            for event in row['events']:
                gold=event['gold'];dst=gold['dst_name'];src=gold['src_name']
                if dst:banks.add(dst.split('/')[0])
                if dst and src:self.assertEqual(dst.split('/')[0],src.split('/')[0])
            self.assertEqual(len(banks),4)
            self.assertEqual(len(row['events'][-1]['gold']['state_after']),16)
if __name__=='__main__':unittest.main()
