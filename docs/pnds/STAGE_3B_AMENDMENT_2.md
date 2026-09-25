# PNDS-GATE-001 Stage 3b — Amendment 2: Learner-Bottleneck Diagnostic

> Registered **before** implementation, following the rule that made the
> Amendment 1 correction possible. This amendment does **not** reopen Stage 3b's
> claim and does **not** modify the environment. It authorises exactly two
> bounded probes whose purpose is to identify the cause of the F1 negative
> result, and nothing else.

## 0. Program state as amended

> F1: **negative** for 70-key sparse-REINFORCE acquisition.
> F2/F3: **positive**. Substrate representability and state intervention remain
> supported. **The learner-level cause of F1 is unresolved.**

That is the correct summary of `dd8f63c`. It is narrower than "Stage 3b
falsified": the substrate is validated and the intervention is validated; what
failed is one specific training signal at one specific key cardinality.

## 1. What the current evidence does and does not establish

| Question | Status | Evidence |
|---|---|---|
| Is the relation representable in the router's basis? | **yes** | `true_key` = 1.0000, margin exactly 2 (`528ad89` §3) |
| Does the state intervention causally matter? | **yes** | `true_key − wrong_key` = 0.8333 (F2 passes) |
| Is the environment free of stateless shortcuts? | **yes** | `t_static`, `static`, `anti_static` all at chance (F3 passes) |
| Can REINFORCE learn a *single* fixed key? | **yes** | 1.0000 at 3000 streams |
| Can REINFORCE learn all 70 keys? | **no** | 0.14–0.21 flat over 1k–16k streams, lr ∈ {0.05, 0.2, 1.0}, 10-seed F1 negative |
| **Is the failure caused by key cardinality?** | **unresolved** | the fixed-key result does not separate cardinality from sparse credit assignment |
| **Is the failure caused by sparse binary credit assignment?** | **unresolved** | no probe has used a dense signal |

The fixed-key 1.0000 result is the load-bearing ambiguity. It proves the learner
*can* optimize this loss surface. It does not tell us whether the failure at 70
keys is "too many tasks for the sample budget" or "too little gradient signal
per sample." Those have different consequences for PNDS, and they are what the
two probes below separate.

## 2. Probe 1 — key-cardinality scaling (k = 2)

### Specification

Identical to the committed Stage 3b protocol in every respect except the key
size. Concretely, **unchanged**: episode generator, exact-`T` constraint,
train/test seed boundary, candidate count, optimizer, the four learning rates,
the 10-seed protocol, the success statistic, the margin statistic, the F1
majority-of-seeds criterion. **Changed**: `k = 2` only, giving `C(8,2) = 28`
keys.

| parameter | value |
|---|---|
| `dim` | 8 |
| `k` | **2** (was 4) |
| keys | `C(8,2) =` **28** (was 70) |
| `T` | `default_T(8, 2) = 2 + 3 = 5` |
| `n_candidates` | 8 |
| train streams | 250, 8 episodes each |
| test streams | 100 per seed |
| seeds | 10, the same seed schedule as §2 of the result record |
| `lr` | 0.05 (primary); 0.20 and 1.00 as the tuning control |

Per-key effective sample size rises from ~28 to ~71 episodes, a factor of 2.5,
with no change to any invariant or control.

### Pre-registered prediction

The credit-assignment account predicts a **monotone dependence on keys-per-update**:
if sparse REINFORCE's failure is that each key block sees too few updates, 28
keys with the same budget should improve, and the 10-seed margin distribution
should shift positive. The pure sample-size account additionally predicts the
improvement is roughly proportional to the 2.5× factor.

**Either outcome is reportable.** A k=2 success does **not** rescue the 70-key
experiment and does **not** reopen the Stage 3b claim; it establishes a
cardinality/credit-assignment *boundary*. A k=2 failure localises the problem
further and makes Probe 2 the decisive test.

### F1 application

F1 applies verbatim: `learned` must beat `learned_no_state` by a margin positive
on a majority of the same 10 seeds, reported with the exact binomial p-value.
The criterion is not relaxed to make k=2 pass, and the result is reported
whichever way it falls.

## 3. Probe 2 — dense-gradient signal (70 keys)

### Specification

Run **only after** Probe 1's result is recorded. The same 70-key environment as
`dd8f63c`, with one change: the training signal.

`dd8f63c` trains with REINFORCE on a scalar binary reward. The gradient reaches
the router as `lr · (r − 1/n) · (1 − p_selected) · φ`, i.e. one scalar bit of
information multiplied across the whole active block. Probe 2 replaces that with
a **dense supervised gradient that identifies the correct key**: the loss is the
cross-entropy of the router's candidate distribution against the gold
distribution, differentiated with respect to the router's weights, computed from
the gold index.

The state read, the basis, the parameter count, the key space, the stream
structure, the train/test boundary and the 10-seed protocol are all unchanged
from `dd8f63c`. Only the source of gradient changes — sparse and outcome-derived
becomes dense and label-derived.

This is a **learner probe**, not a new PNDS mechanism. The gold index is used
only to compute the gradient; it is never given to the router at test time, and
the state is still read only through the routing interface.

### Pre-registered decision table

| Probe 1 (k=2, sparse) | Probe 2 (70-key, dense) | Interpretation |
|---|---|---|
| succeeds | succeeds | sparse acquisition scales poorly with key cardinality; the failure is sample-allocation, and dense signal recovers it |
| succeeds | fails | deeper optimization or representation problem, not credit assignment |
| fails | succeeds | **strong evidence that sparse binary credit assignment is the bottleneck** |
| fails | fails | the learner problem is broader than sparse REINFORCE; the substrate/learner separation itself needs revisiting |

The row that matters for PNDS is the third. If it lands there, the conclusion is
that PNDS persistent state is learnable *given a sufficiently informative
training signal*, which is a constraint on the training loop rather than on the
architecture — and it changes what Stage 4 should test.

## 4. Methodological constraints — binding

These are the constraints the user set, restated so a later reader (or a later
agent) cannot drift past them:

1. **No architectural change.** No recurrence, no memory, no attention over
   streams, no key-prediction head, no auxiliary task.
2. **No new components.** No verifier, no critic, no reward shaping, no
   curriculum, no contrastive loss, no self-supervision.
3. **No change to the state representation.** `PersistentState` stays exactly
   as committed in `dd8f63c`.
4. **No environment redesign.** The generator, the exact-`T` construction, the
   candidate sampler and the controls are frozen. Probe 1 changes `k` only;
   Probe 2 changes the training signal only.
5. **No additional tuning.** The learning rates are the three already tested.
   No search over temperature, batch size, or stream length.
6. **The retraction stands.** The basis-dependent claims "data starvation",
   "7× negative-gradient interference" and "supervised ceiling at chance" are
   **not findings** and are not revived by either probe. If a probe's result
   superficially resembles one of them, the resemblance is noted and the
   retraction is restated, not repealed.
7. **Both probes report F1 with the same 10-seed protocol and the exact
   binomial p-value**, and neither is re-run with a different seed schedule
   after the fact.

The current negative result is valuable *precisely because* the substrate has
been isolated from the learner. The next experiment identifies the learner
bottleneck; it does not make the architecture more complicated.

## 5. What this amendment does not authorize

- Re-opening the Stage 3b claim under a modified protocol.
- Any change to `STAGE_3B_RESULT.md`'s numbers. That document is a closed
  record at `dd8f63c`; the probes get their own result documents.
- Promotion of any PNDS claim from either probe alone. If Probe 2 succeeds at
   70 keys with a dense signal, that is evidence about the *training loop*, not
   about the substrate, and it is graded accordingly.
- Any change to the property tests of `2a53338`. Both probes must continue to
  pass them.

## 6. Implementation order

1. This amendment is committed **before** any probe code is written.
2. Probe 1 (`k=2`) is implemented, run, and its result recorded as
   `STAGE_3B_PROBE1_RESULT.md`.
3. **Only then** is Probe 2 implemented, run, and recorded as
   `STAGE_3B_PROBE2_RESULT.md`.
4. The two results are interpreted jointly through the table in §3.

No probe is begun before the previous one's document is committed.
