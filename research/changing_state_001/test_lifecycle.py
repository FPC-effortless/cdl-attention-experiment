import random
import unittest
from lifecycle import CurrentState, Event, history_query

class LifecycleTests(unittest.TestCase):
    def check_trace(self, events, keys=8):
        state = CurrentState()
        prefix = []
        for event in events:
            state.apply(event)
            prefix.append(event)
            for key in range(keys):
                self.assertEqual(state.query(key), history_query(prefix, key))
        return state

    def test_snapshot_and_live_reference(self):
        s = self.check_trace([Event('SET',0,2),Event('COPY',1,0),Event('LINK',2,0),Event('SET',0,7)])
        self.assertEqual([s.query(k) for k in range(3)], [7,2,7])

    def test_delete_recreate_dangling_and_unknown_copy(self):
        s = self.check_trace([Event('LINK',1,0), Event('COPY',2,0),Event('SET',0,3),Event('DELETE',0),Event('SET',0,6)])
        self.assertEqual([s.query(k) for k in range(3)], [6,6,None])

    def test_cycles_break_and_self_copy(self):
        s=self.check_trace([Event('LINK',0,1),Event('LINK',1,0),Event('COPY',2,0),Event('SET',1,7),Event('COPY',0,0),Event('SET',1,3)])
        self.assertEqual([s.query(k) for k in range(3)], [7,3,None])

    def test_untouched_records_and_pure_queries(self):
        s=CurrentState()
        s.apply(Event('SET',0,2)); s.apply(Event('SET',1,5))
        before=dict(s.records)
        for _ in range(5): self.assertEqual(s.query(0),2)
        self.assertEqual(s.records,before)
        s.apply(Event('SET',1,6)); self.assertEqual(s.records[0],before[0])

    def test_generated_differential_prefixes(self):
        for seed in range(64):
            rng=random.Random(seed)
            self.check_trace([Event(rng.choice(['SET','COPY','LINK','DELETE','NOP']),rng.randrange(8),rng.randrange(8)) for _ in range(64)])

    def test_reject_bad_events(self):
        for args in [('BAD',0,0),('SET',True,0),('SET',0,-1),('SET',0,float('nan'))]:
            with self.assertRaises(ValueError): Event(*args)

if __name__ == '__main__': unittest.main()
