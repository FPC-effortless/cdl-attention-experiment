"""Deterministic arithmetic and bounded-index fixtures; no learned-model claims."""
from fractions import Fraction
import importlib.util
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('audited_paged_index',
    ROOT / 'relational_composition/hierarchical_index.py')
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def preceding_match(data, t, k, window=512):
    """Newest prior continuation with matching preceding k bytes, or None.

    Selection reads only data[:t]. Whether its continuation equals data[t] is
    an evaluation label and is deliberately not returned by this function.
    """
    if not 0 <= t <= len(data) or k < 1 or window < k:
        raise ValueError('invalid position, order or window')
    if t < k:
        return None
    query = data[t-k:t]
    for j in range(t-1, max(k, t-window)-1, -1):
        if data[j-k:j] == query:
            return j
    return None


def run():
    fixture = []
    for ordered_keys in itertools.permutations(('A', 'B')):
        index = module.PagedCausalIndex(2)
        semantic = {}
        for position, key in enumerate(ordered_keys):
            index.append(position, 0, 0)  # oracle address, same bucket
            semantic[position] = key
        candidates = index.lookup_page(0, 0, 2, 0).candidates
        assert len(candidates) == 2
        for query in ('A', 'B'):
            blind = candidates[0]
            matches = [j for j in candidates if semantic[j] == query]
            assert len(matches) == 1
            fixture.append(dict(order=ordered_keys, query=query,
                candidate_admitted=True, blind_correct=semantic[blind] == query,
                verified_correct=semantic[matches[0]] == query))
        assert not any(semantic[j] == 'missing' for j in candidates)
    assert sum(r['blind_correct'] for r in fixture) == 2
    assert all(r['verified_correct'] for r in fixture)

    paging = 0
    for cap in (1, 2, 3):
        index = module.PagedCausalIndex(cap)
        for pos in range(20):
            index.append(pos, pos % 2, 0)
        for t in range(21):
            for rel in (0, 1):
                expected = [j for j in range(t-1, -1, -1) if j % 2 == rel]
                for rank, identity in enumerate(expected):
                    assert index.lookup_ordinal(rel, 0, t, rank) == identity
                    paging += 1
                assert not index.lookup_page(rel, 0, t, len(expected)//cap + 1).candidates

    for k in (3, 4, 6):
        context = b'abcdef'[:k]
        data = context + b'X' + context + b'X'
        t = len(data) - 1
        j = preceding_match(data, t, k)
        assert j == k and data[j] == data[t]
        changed = data[:-1] + b'Y'
        assert preceding_match(changed, t, k) == j and changed[j] != changed[t]
        assert preceding_match(data + b'future', t, k) == j

    base, cache, gate = [Fraction(1, 4), Fraction(3, 4)], [Fraction(1), Fraction(0)], Fraction(1, 4)
    mixed = [(1-gate)*p + gate*q for p, q in zip(base, cache)]
    mutant = [p + gate*q for p, q in zip(base, cache)]
    assert sum(mixed) == 1 and sum(mutant) != 1
    A, H = Fraction(6211,10000), Fraction(9567,10000)
    return dict(status='PASS', kind='mechanical fixtures, NOT learned neural evaluation',
        same_bucket_cases=fixture, paging_checks=paging,
        mixture_normalization=True, wrong_weight_mutation_detected=True,
        ngram_orders=[3,4,6], ngram_causality_checks=True,
        reported_number_derivations=dict(
            conditional_answer_given_hit_bounds=[float((A-(1-H))/H),float(A/H)],
            random_slot_success_with_no_miss_recovery=float(H/2),
            implied_seen_byte_fraction=float(Fraction(211+69,145+211))))


if __name__ == '__main__':
    result = run()
    if len(sys.argv) == 2:
        Path(sys.argv[1]).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
