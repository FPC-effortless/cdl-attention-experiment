"""Deterministic engineering checks, not training or quality evaluation."""
import hashlib
import json
import platform
from pathlib import Path

import numpy as np

from reference import Config, counts, decay, initialize, parameters, run, step


def equal_base(a, b):
    return (np.array_equal(a.values, b.values) and np.array_equal(a.links, b.links)
            and a.cursor == b.cursor and a.position == b.position
            and np.array_equal(a.older, b.older) and np.array_equal(a.recent, b.recent))


def main():
    c = Config()
    counts_actual = sum(x.size for x in parameters(c, 0).values())
    assert counts_actual == counts(c)["candidate_parameters"]
    counter = {"zero_injection_sequences": 0, "bounded_steps": 0,
               "prefix_checks": 0, "frame_checks": 0, "padding_checks": 0,
               "reset_checks": 0, "independent_batch_checks": 0}
    max_value = max_trace = 0.0
    for seed in range(6):
        rng = np.random.default_rng(seed + 1000)
        tokens = rng.integers(0, c.vocabulary, 96).tolist()
        p = parameters(c, seed)
        base = Config(traces=0)
        bp = parameters(base, seed)
        zero = {k:v.copy() for k,v in p.items()}
        zero["trace_to_controller"].fill(0)
        for mode in ("rewrite", "ring_write", "no_links"):
            ref_state, ref_out = run(base, bp, tokens, decay(base, "instant"), mode)
            off_state, off_out = run(c, zero, tokens, decay(c, "multi"), mode)
            assert equal_base(ref_state, off_state)
            assert np.array_equal(ref_out, off_out)
            counter["zero_injection_sequences"] += 1
            for variant in ("instant", "single", "multi"):
                lam = decay(c, variant)
                st = initialize(c)
                for tok in tokens:
                    prev = st
                    saved = prev.values.copy(), prev.links.copy(), prev.regulation.copy()
                    st, logits, tr = step(c, p, st, tok, lam, mode)
                    assert np.array_equal(prev.values, saved[0])
                    assert np.array_equal(prev.links, saved[1])
                    assert np.array_equal(prev.regulation, saved[2])
                    keep = np.arange(c.slots) != tr["destination"]
                    assert np.array_equal(st.values[keep], prev.values[keep])
                    if not tr["admitted"]:
                        assert np.array_equal(st.values, prev.values)
                    assert st.cursor == tr["destination"]
                    assert np.array_equal(st.values[st.cursor], tr["written"])
                    assert st.position == prev.position + 1
                    assert np.max(np.abs(st.values)) <= 1 + 1e-12
                    assert np.max(np.abs(st.regulation)) <= 1 + 1e-12
                    assert np.isfinite(logits).all()
                    max_value = max(max_value, float(np.abs(st.values).max()))
                    max_trace = max(max_trace, float(np.abs(st.regulation).max()))
                    counter["bounded_steps"] += 1
                    counter["frame_checks"] += 1
                prefix_state, prefix_logits = run(c, p, tokens[:37], lam, mode)
                _, full_logits = run(c, p, tokens, lam, mode)
                assert np.array_equal(prefix_logits, full_logits[:37])
                counter["prefix_checks"] += 1
                padded, out, _ = step(c, p, prefix_state, 0, lam, mode, valid=False)
                assert padded is prefix_state and out is None
                counter["padding_checks"] += 1
                reset, output, _ = step(c, p, prefix_state, tokens[0], lam, mode, reset=True)
                fresh, freshout, _ = step(c, p, initialize(c), tokens[0], lam, mode)
                assert equal_base(reset, fresh) and np.array_equal(reset.regulation, fresh.regulation)
                assert np.array_equal(output, freshout)
                counter["reset_checks"] += 1
                # Interleave two streams to detect accidental shared mutable state.
                sa, sb = initialize(c), initialize(c)
                oa, ob = [], []
                other = tokens[::-1]
                for a,b in zip(tokens, other):
                    sa, ya, _ = step(c,p,sa,a,lam,mode)
                    sb, yb, _ = step(c,p,sb,b,lam,mode)
                    oa.append(ya); ob.append(yb)
                _, alonea = run(c,p,tokens,lam,mode)
                _, aloneb = run(c,p,other,lam,mode)
                assert np.array_equal(np.stack(oa), alonea)
                assert np.array_equal(np.stack(ob), aloneb)
                counter["independent_batch_checks"] += 1
    nonfinite = parameters(c, 1)
    nonfinite["embedding"][0,0] = np.nan
    try:
        step(c,nonfinite,initialize(c),0,decay(c,"multi"))
    except ValueError:
        nonfinite_rejected = True
    else:
        raise AssertionError("Nonfinite-domain contract was not enforced")
    result = {"status":"PASS", "kind":"finite deterministic engineering fixtures",
              "training_runs":0, "claims_excluded":["language quality","learning convergence",
              "PyTorch refinement","runtime speedup","formal proof of floating arithmetic"],
              "checks":counter, "largest_abs_record":max_value, "largest_abs_trace":max_trace,
              "nonfinite_input_rejected":nonfinite_rejected,
              "actual_allocated_parameters":counts_actual,"accounting":counts(c),
              "python":platform.python_version(),"numpy":np.__version__,
              "source_sha256":{f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
                               for f in ("reference.py","verify_reference.py")}}
    target=Path("verification/reference.json")
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
