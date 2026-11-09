# CSM-MATH-001 — mathematical engineering of persistent regulation

This research package adds a small token-driven regulatory state to the existing
Compositional State Model (CSM). It specifies the transition, supplies an indexed
NumPy reference, derives the relevant invariants and counterexamples, and checks
structural properties in Lean. **No model was trained; no accuracy or latency
improvement is established.**

The inspected baseline is commit
`3edc4f1a541a3221390e4c0a21fe0c12004f1ca5` of
`FPC-effortless/cdl-attention-experiment`. The original source archives,
checkpoints and prior results are unchanged. The reference is a fresh
transcription of the documented equations, not a verified refinement of the
archived PyTorch implementation.

## Candidate

Before a valid token, the state consists of record values, one link per record,
the cursor, the previous two **input embeddings**, token position and a new
regulator vector. Given the original causal frontend feature `f`:

```text
s      = tanh(B f + b)
r_next = lambda * r + (1 - lambda) * s
u      = tanh(C [f; record_at_cursor] + U r_next)
```

All original reads, operation selection, proposal, value admission/gain, link
update, cursor update and output projection remain unchanged. A token produces
the prediction of the following token. There are no target-derived forward
signals. The proposed initial setting uses 12 coordinates: four each at
`tau = 4, 32, 256` observed tokens with `lambda = exp(-1/tau)`. The decays are
fixed, illustrative choices. A relaxation time is not a half-life; the half-life
is `tau * log(2)`.

Padding is identity and emits no logits. Reset occurs before a valid token and
resets the entire state; reset on padding is rejected. Chunk boundaries carry
state unless explicitly marked as document resets. The initial regulator is
zero, so slow channels have a real cold-start transient.

## Mathematical conclusions

- Setting `U = 0` recovers the baseline **core** trajectory and readout with the
  same original weights and inputs. The unused added regulator may still evolve.
- Bounded tanh proposals and convex gains preserve record and regulator
  coordinate bounds in exact real arithmetic. This does not establish global
  contraction, useful memory, or floating-point refinement.
- Token-only trace inputs permit affine prefix scan, using chronological
  composition `(a1,b1) then (a2,b2) = (a2*a1, a2*b1+b2)`. The original record /
  controller feedback remains sequential. Real associativity is not bitwise
  floating-point associativity or a speed measurement.
- A state-dependent write gain contributes `(z-w) grad(alpha)^T` to the
  Jacobian. Dropping this term is incorrect. The package includes bounded
  counterexamples to general contraction and equilibrium-stability claims.
- The sufficient contraction criterion is local to an admitted, fixed-route,
  active-record subsystem with other read records fixed. Untouched records
  prevent strict one-step contraction of the entire bank.
- Finite real sigmoid writes retain a vector residual. Correct discrete
  decoding and exact vector replacement are distinct contracts.
- Distinct write destinations suffice to commute **frozen-snapshot commits**;
  recomputed transactions also need compatible read footprints. Complete token
  steps do not acquire a commutation guarantee.

See `specification.tex` for equations, proofs, gradient derivation and the
matched-capacity comparison. `independent_review.txt` and `trace_review.txt`
record the mathematical reviews; `verification/final_review.txt` records the
four final wording corrections incorporated into the specification.

## Recorded verification

| Evidence | Actual result | Boundary |
| --- | --- | --- |
| Reference fixtures | 18 zero-injection comparisons; 5,184 bounded/frame steps; 54 each of prefix, padding, reset and independent-stream checks | Finite float64 cases, no archived-runtime parity |
| Exact trace checks | 3,375 associativity triples, 120 closed-form/impulse checks, 109 scan trials covering 1,397 state vectors | Exact fractions, not measured FP32 kernels |
| Smooth gradients | 64 coordinate comparisons in four cases; maximum absolute error `4.038687023699694e-10` | Regulator/tanh-controller subgraph with frozen base input; no hard-route estimator validation |
| Stability examples | 14 deterministic checks passed | Analytic corroboration and counterexamples; no trained-weight certificate |
| Lean | 23 declared theorems freshly compiled at trust level 0; axiom audit; two false controls rejected | Structural transitions and affine algebra under explicit laws; no new real-analysis proof |

The audited theorem dependencies were only `propext` and `Quot.sound`. No
`sorryAx`, added axioms or native proof oracle appeared. The normal official Lean
executable could not locate itself in this runtime. The previously disclosed CSM
launcher supplies its explicit application path to the unchanged official
`lean_main` shared library. This is an adapted launch, not a second independent
kernel. Commands, source/toolchain hashes, axiom output and negative-control
results are preserved in `verification/lean.json` and `verification/axioms.log`.

Early development failures and their fixes are recorded in
`verification/development_notes.txt`; the final status is not presented as an
error-free first attempt.

## Cost accounting

For `S=8, d=32, operations=4, hidden=64, vocabulary=19, regulator=12`:

| Quantity | Baseline | Candidate | Addition |
| --- | ---: | ---: | ---: |
| Learned parameters, independently allocated and counted | 18,294 | 19,074 | 780 |
| Nominal dense MAC terms per token | 18,560 | 19,328 | 768 |
| Retained FP32 tensor bytes per stream, original dense convention | 1,568 | 1,616 | 48 |

The added formulas are `2*d*m + m` parameters, `2*d*m` nominal MAC terms, and
`4*m` FP32 state bytes. Counts exclude elementwise work, routing overhead,
bandwidth, allocation, backward/optimizer work and the position integer. The
NumPy reference uses float64 and indexed integer links, so the dense-source
cache/MAC figures are not its actual allocation or instruction counts.

## Reproduce

Recorded runtime: Python 3.12.14, NumPy 2.3.5, Lean 4.19.0. The exact trace
checker uses only the Python standard library. The Lean source imports `Std`,
not Mathlib. Work in a fresh checkout if preserving the original receipts.

```bash
python3 -m pip install -r requirements.txt
python3 verify_reference.py
python3 trace_algebra.py > verification/trace_algebra.json
python3 verify_stability.py
python3 verify_trace_gradients.py
python3 verify_lean.py --lean /path/to/lean
```

If the same application-path discovery failure occurs, the explicit-path
launcher can be rebuilt against the official distribution. The following
assumes that distribution was extracted to `/tmp/lean-4.19.0-linux`:

```bash
cc -O2 -std=c11 -Wl,--export-dynamic tooling/lean_explicit_path.c \
  -L /tmp/lean-4.19.0-linux/lib/lean \
  -Wl,-rpath,/tmp/lean-4.19.0-linux/lib/lean \
  -lleanshared -ldl -o /tmp/csm-lean-explicit-path
python3 verify_lean.py \
  --lean /tmp/lean-4.19.0-linux/bin/lean \
  --explicit-path-launcher /tmp/csm-lean-explicit-path \
  --distribution-archive /tmp/csm-lean-4.19.0-linux.tar.zst
```

The official release URL and measured archive hash are in `provenance.json`.
The archive hash records the downloaded bytes; it is not a separate publisher
signature verification. The source manifest hashes the delivered text and
receipts; it deliberately excludes itself and generated build intermediates.

Build the report with two LaTeX passes for the table of contents:

```bash
pdflatex -interaction=nonstopmode -halt-on-error specification.tex
pdflatex -interaction=nonstopmode -halt-on-error specification.tex
```

## Next empirical decision

Compare the original model with immediate, single-timescale and multi-timescale
regulation. The latter three have the same learned-array shapes; the single and
multi variants have the same carried-state capacity. The immediate variant may
omit the carried trace at deployment. Hold supervision, data, optimizer,
initialization policy and update budget fixed. A zero `U` initialization gives
baseline inclusion but initially blocks task gradients to `B,b`; document that
optimization choice.

Use trace-reset/swap interventions to test whether the state is causally used.
Report per-seed control failures and actual end-to-end inference cost. Restricted
SET/recall success is a preservation check; it cannot establish broad reasoning,
language competence or architectural novelty. No training campaign is included
in this package.
