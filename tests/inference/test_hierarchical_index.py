import random
import unittest
from relational_composition.hierarchical_index import (
    PagedCausalIndex, rank_address, rank_bits, bits_page)
from relational_composition.index import CausalIndex


class HierarchicalIndexTests(unittest.TestCase):
    def test_complete_accessibility_collision_saturation(self):
        ix=PagedCausalIndex(cap=2)
        for i in range(10017):
            ix.append(i,0,0)
        for rank in range(10017):
            page,slot=rank_address(rank,2)
            self.assertEqual(ix.lookup_page(0,0,10017,page).candidates[slot],10016-rank)
        self.assertEqual(ix.page_count(0,0,10017),5009)
        self.assertEqual(ix.lookup_page(0,0,10017,5009).candidates,())
        self.assertEqual(ix.lookup_page(0,0,10017,0).candidates,(10016,10015))
        self.assertEqual(ix.lookup_page(0,0,5,0).candidates,(4,3))
        self.assertEqual(ix.lookup_page(0,0,5,1).candidates,(2,1))
        self.assertEqual(ix.lookup_page(0,0,5,2).candidates,(0,))
        self.assertEqual(ix.lookup_page(0,0,0,0).candidates,())

    def test_random_causality_document_keys_and_recency_equivalence(self):
        rng=random.Random(313)
        for _ in range(20):
            cap=rng.randrange(1,6)
            paged=PagedCausalIndex(cap)
            old=CausalIndex(cap)
            writes=[]
            for t in range(100):
                key=(rng.randrange(3),rng.randrange(3))
                writes.append(key)
                paged.append(t,*key)
                old.append(t,*key)
            for _ in range(300):
                t=rng.randrange(101)
                key=(rng.randrange(3),rng.randrange(3))
                read=paged.lookup_page(*key,t,0)
                self.assertEqual(read.candidates,old.lookup(*key,t).candidates)
                self.assertEqual(read.omitted,old.lookup(*key,t).omitted)
                self.assertLessEqual(len(read.candidates),cap)
                self.assertTrue(all(j<t and writes[j]==key for j in read.candidates))
                eligible=[j for j in range(t) if writes[j]==key][::-1]
                reconstructed=tuple(x for page in range(read.pages)
                    for x in paged.lookup_page(*key,t,page).candidates)
                self.assertEqual(reconstructed,tuple(eligible))

    def test_codes_and_fail_closed(self):
        for width in (1,3,12,24):
            for p in [0,1,3,15,42,501] if width>=10 else range(1<<width):
                if p >= (1<<width):
                    continue
                self.assertEqual(bits_page(rank_bits(p,width)),p)
        with self.assertRaises(ValueError):
            rank_address(-1,2)
        with self.assertRaises(ValueError):
            bits_page([2])
        with self.assertRaises(ValueError):
            PagedCausalIndex(0)
        ix=PagedCausalIndex(2)
        ix.append(0,1,1)
        with self.assertRaises(ValueError):
            ix.append(0,1,1)
        with self.assertRaises(ValueError):
            ix.lookup_page(1,1,1,-1)

if __name__=='__main__':
    unittest.main()
