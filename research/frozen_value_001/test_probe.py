import unittest
from run import generate,encode,VOCAB,PANELS,replay,history_map

class ProbeTests(unittest.TestCase):
    def test_complete_grid_and_labels(self):
        rows=generate();self.assertEqual(len(rows),128)
        self.assertEqual(len({r['record_id'] for r in rows}),128)
        self.assertEqual(len({r['subject_id'] for r in rows}),1)
        for panel in PANELS:
            part=[r for r in rows if r['observation']['probe']['panel']==panel]
            self.assertEqual({(r['observation']['probe']['query'],r['observation']['probe']['expected']) for r in part},
                             {(f'r{k}',f'v{v}') for k in range(4) for v in range(8)})
        for r in rows:
            m=r['observation']['memory'];p=r['observation']['probe']
            self.assertEqual(replay(m),history_map(m));self.assertEqual(replay(m)[p['query']],p['expected'])

    def test_matched_context_and_old_value_is_unique(self):
        rows={r['record_id']:r for r in generate()}
        for key in range(4):
            for value in range(8):
                cases=[rows[f'{p}-r{key}-v{value}']['observation'] for p in PANELS]
                backgrounds=[e for e in cases[1]['memory']['events'] if e['key']!=f'r{key}']
                self.assertEqual(len({e['value'] for e in backgrounds}),3)
                self.assertNotIn(cases[1]['probe']['old'],{e['value'] for e in backgrounds})
                self.assertNotIn(cases[1]['probe']['expected'],{e['value'] for e in backgrounds})
                for c in cases[2:]:
                    self.assertEqual([e for e in c['memory']['events'] if e['key']!=f'r{key}'],backgrounds)
                    self.assertEqual(replay(c['memory']),replay(cases[1]['memory']))

    def test_encoding_no_labels_in_token_inputs(self):
        for row in generate():
            m=row['observation']['memory'];p=row['observation']['probe'];x=encode(m,p['query'])
            self.assertEqual(len(x['ids']),27)
            self.assertEqual(x['ids'][-3:],[VOCAB['ask'],VOCAB[p['query']],VOCAB['_']])
            self.assertEqual(x['destination'][-1],7)
            self.assertEqual(x['second_address'][-1],-100)
            self.assertEqual(x['link_edit'][-1],0)

if __name__=='__main__':unittest.main()
