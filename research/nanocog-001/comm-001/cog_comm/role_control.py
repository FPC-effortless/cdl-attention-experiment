"""COG-COMM-ROLE-001: fresh, synthetic local role-binding preflight.

No historical files or protected splits are inputs. Each family forms a full
2^3 crossing, not eight independent observations. This is familiar grammar
and vocabulary with disjoint name-pair/color-pair combinations, not language
generalization. A model gets only the causal prompt, never the row metadata.
"""
import itertools
import json
import math
import random
import sys
import unittest
from collections import Counter

NAMES = tuple("ada ben cy di eli fay gio hal ivy jay kai lou mia nik ola pam".split())
COLORS = tuple("blue gold green orange pink purple red white".split())
CELLS = tuple(itertools.product((-1, 1), repeat=3))
WORDS = ("is",) + NAMES + COLORS
ARMS = (
    "true_role", "opposite_role", "first_clause", "last_clause",
    "fixed_left", "fixed_right", "fixed_color_a", "fixed_color_b",
)

def families():
    pool = list(itertools.product(itertools.combinations(NAMES, 2),
                                  itertools.combinations(COLORS, 2)))
    random.Random(20261122).shuffle(pool)
    return {"fit": pool[:320], "control": pool[320:440]}

def world(key, cell):
    (l, r), (a, b) = key
    s, o, q = cell
    v, w = (a, b) if s == 1 else (b, a)
    clauses = ((l, "is", v), (r, "is", w))
    if o == -1:
        clauses = clauses[::-1]
    return clauses[0] + clauses[1] + (l if q == 1 else r,)

def oracle(prompt):
    l, sep1, a, r, sep2, b, asked = prompt
    if (sep1, sep2) != ("is", "is") or l == r or a == b:
        raise ValueError("invalid clauses")
    if l not in NAMES or r not in NAMES or a not in COLORS or b not in COLORS:
        raise ValueError("invalid vocabulary")
    return {l: a, r: b}[asked]

def templates(cell):
    s, o, q = cell
    return (s*q, -s*q, s*o, -s*o, s, -s, 1, -1)

def score(key, predictions):
    if set(predictions) != set(CELLS):
        raise ValueError("expected exactly eight cells")
    a, b = key[1]
    signs, full, margins = {}, {}, {}
    for cell in CELLS:
        row = predictions[cell]
        if set(row) != set(WORDS) or not all(math.isfinite(x) for x in row.values()):
            raise ValueError("need one finite logit for every vocabulary item")
        y = oracle(world(key, cell))
        margins[cell] = row[a] - row[b]
        if not math.isfinite(margins[cell]):
            raise ValueError("nonfinite margin")
        signs[cell] = (margins[cell] > 0) - (margins[cell] < 0)
        full[cell] = row[y] > max(v for k, v in row.items() if k != y)
    agreement = {arm: sum(signs[c] == templates(c)[i] for c in CELLS)
                 for i, arm in enumerate(ARMS)}
    sq, so, s0, bias = (sum(margins[c] * templates(c)[i] / 8 for c in CELLS)
                        for i in (0, 2, 4, 6))
    return dict(two_color=agreement["true_role"], strict_family=agreement["true_role"] == 8,
                full_vocab=sum(full.values()), full_family=all(full.values()),
                query_both_correct=sum(all(signs[s,o,q] == s*q for q in (-1,1))
                                       for s,o in itertools.product((-1,1), repeat=2)),
                order_both_correct=sum(all(signs[s,o,q] == s*q for o in (-1,1))
                                       for s,q in itertools.product((-1,1), repeat=2)),
                ties=sum(y == 0 for y in signs.values()), agreement=agreement,
                canonical_margin_contrasts=dict(role=sq, first=so, left=s0, bias=bias))

def fixture(key, arm):
    out = {}
    a, b = key[1]
    for c in CELLS:
        sign = templates(c)[ARMS.index(arm)]
        out[c] = dict.fromkeys(WORDS, -2.0)
        out[c].update({a: float(sign), b: float(-sign)})
    return out

class SelfTest(unittest.TestCase):
    def test_splits(self):
        f = families()
        self.assertEqual({k: len(set(v)) for k,v in f.items()}, {"fit":320,"control":120})
        self.assertFalse(set(f["fit"]) & set(f["control"]))
        for keys in f.values():
            for j, expected in enumerate((NAMES,COLORS)):
                self.assertEqual({x for key in keys for x in key[j]}, set(expected))

    def test_one_factor_edges(self):
        for group in families().values():
            for key in group:
                self.assertEqual(len({world(key,c) for c in CELLS}), 8)
                for s,o,q in CELLS:
                    p = world(key,(s,o,q))
                    self.assertEqual(len(p), 7)
                    self.assertLessEqual(set(p), set(WORDS))
                    self.assertEqual(p[:-1], world(key,(s,o,-q))[:-1])
                    self.assertNotEqual(oracle(p), oracle(world(key,(s,o,-q))))
                    self.assertEqual(p[3:6]+p[:3]+p[6:], world(key,(s,-o,q)))
                    self.assertEqual(oracle(p), oracle(world(key,(s,-o,q))))
                    self.assertEqual(p[:2]+(p[5],)+p[3:5]+(p[2],)+p[6:], world(key,(-s,o,q)))
                    self.assertNotEqual(oracle(p), oracle(world(key,(-s,o,q))))
                    self.assertEqual(oracle(p), key[1][int(s*q == -1)])

    def test_shortcuts(self):
        for key in families()["control"]:
            for arm, expected in zip(ARMS, (8,0,4,4,4,4,4,4)):
                result = score(key, fixture(key, arm))
                self.assertEqual(result["two_color"], expected)
                self.assertEqual(result["strict_family"], expected == 8)
                self.assertEqual(result["agreement"][arm], 8)

    def test_failure_paths(self):
        key = families()["control"][0]
        scores = fixture(key, "true_role")
        a,b = key[1]
        c = CELLS[0]
        scores[c][a] = scores[c][b] = 0.0
        self.assertEqual(score(key, scores)["ties"], 1)
        scores = fixture(key, "true_role")
        scores[c]["is"] = 100.0
        result = score(key, scores)
        self.assertTrue(result["strict_family"])
        self.assertFalse(result["full_family"])
        scores[c]["is"] = float("nan")
        with self.assertRaises(ValueError):
            score(key, scores)
        del scores[c]
        with self.assertRaises(ValueError):
            score(key, scores)

def report():
    hold = families()["control"]
    return {a: dict(worlds=sum(score(k,fixture(k,a))["two_color"] for k in hold),
                    families=sum(score(k,fixture(k,a))["strict_family"] for k in hold))
            for a in ARMS}

if __name__ == "__main__":
    if sys.argv[1:] == ["--fixture-report"]:
        print(json.dumps({"scope":"deterministic software fixtures, not learned models",
                          "worlds":960,"families":120,"arms":report()}, sort_keys=True))
    else:
        unittest.main()
