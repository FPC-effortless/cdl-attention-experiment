"""Check counterfactual labels against the task parser and reconcile saved measurements."""
import json
from pathlib import Path
import sys

from trace_fault import cases
from repair_objective import trace_targets


def verify(out):
    grids = 0
    for domain in ("symbol", "english"):
        qs = ("#a00=", "#b00=") if domain == "symbol" else tuple(
            f"What digit does the log give for {n}?" for n in ("Ada","Ben"))
        for independent, train_seed, test_seed in ((False,31091,48153),(True,861359,957427)):
            train = list(cases(384,train_seed,False,domain,independent))
            test = list(cases(192,test_seed,True,domain,independent))
            train_histories = {h for a,b,_,_ in train for h in (a,b)}
            assert not any(h in train_histories for a,b,_,_ in test for h in (a,b))
            for a,b,i,j in test:
                assert i != j and sorted(a) == sorted(b)
                for history, expected in ((a,(i,j)),(b,(j,i))):
                    for question, digit in zip(qs, expected):
                        assert trace_targets((history+question).encode())[1] == 48+digit
                grids += 1
    for seed in (9731,9743,9769):
        def load(label):
            return json.loads((out/f"{label}.json").read_text())
        d = load(f"C-{seed}")
        e = load(f"binding-C-balanced-trace_pair-{seed}")
        f = load(f"english-independent-{seed}")
        q = load(f"query-{seed}")
        assert d["checkpoint_sha256"] == e["checkpoint_sha256"] == f["checkpoint_sha256"] == q["checkpoint_sha256"]
        for domain, binding in (("symbol",e),("english",f)):
            assert d[domain]["chunk_parity_max"] < 0.0001
            assert binding["head_correct"] == q[domain]["delimiter"]["correct"]
            assert binding["correct_grids"] == q[domain]["delimiter"]["grid_correct"]
        assert set(load(f"grad-{seed}")["groups"]) == {"all","local","blocks","embedding"}
    print(f"{grids} held grids validated against the original task parser; split disjointness, output parity and three checkpoint identities verified.")


if __name__ == "__main__":
    verify(Path(sys.argv[1]))
