# Mathematical engineering and formal verification review

2026-11-08. Reviewed repository baseline:
`FPC-effortless/cdl-attention-experiment`, PR #51,
`b50e1615e16410b06af34081e83762460c0f9de3`.

## Verdict and evidence boundary

Prioritize same-bucket disambiguation before another routing curriculum. The
user's cautious diagnosis is justified; the stronger paragraph claiming that the
ladder **proves credit assignment is the cause** is not. Successful supervised
routing establishes capacity under that supervision. It does not isolate the
learning signal from imposed address targets, optimization, or soft/hard mismatch.

The cited 27.57%, 26.33%, 25.39%, 62.11%, 95.67%, byte-cache scores and successful
Lean mutation were not found in the current accessible repository, Library
inventory/search, or retrieved conversation context. They are user-reported
observations for this review, not independently reproduced measurements. The
available sources still mark Lean as unverified and contain no byte-mixture
implementation. The three pointer-ladder seed values cannot be reconstructed
from their mean; none are invented here. The newer bundle/commit and raw seed
results are required to review that implementation or run its trained readout.

## 1. Candidate admission, selection and decoding are separate

Let H mean that the gold occurrence is admitted, and Y mean the answer is right.
For metrics computed on the same examples, the exact accounting identity is

`P(Y) = P(H) P(Y | H) + (1-P(H)) P(Y | not H)`.

Using the reported rounded values gives
`0.603951 <= P(Y | H) <= 0.649211`, because the unknown miss-case accuracy lies
between zero and one. Thus at least **35.08% of hit cases still fail**. The upper
endpoint 64.92% assumes no correct answer on a miss; dividing accuracy by hit rate
without that qualification is not an exact conditional measurement. This does
not isolate selection from decoding: log both separately.

With exactly two candidates and exactly one correct occurrence, random **slot
selection conditional on admission** succeeds 50% of the time. Without any miss
recovery its aggregate gold-selection rate would be 47.835%, not 50%. Answer
accuracy need not equal selection accuracy when candidate values duplicate, the
decoder fails, or a local shortcut guesses correctly. Require distinct values,
report candidate counts, and retain the 4.33% admission misses as a separate cell.

### Code-grounded finding

In `hierarchical_model.py`, `rank_first(first)` is evaluated before `j1` is
retrieved; `first` contains query state, anchor and relation embedding, but no
candidate key. `rank_second(second)` sees the first operand but does not inspect
the second read's candidate keys. `expressive_model.py` has the same ordering.
There is no explicit key-verification stage in either bounded candidate read.

If candidate order is independently shuffled while the rank head's inputs remain
fixed, that head cannot identify which of two positions contains the matching
semantic key above 50% over balanced permutations. This statement concerns the
blind selection head, not the complete multi-layer model, which may obtain extra
information through other reads. Formal address equality only checks learned
integer codes; it does not imply equality of semantic entities.

Paging and key verification address different problems. Paging exposes old
records; verification distinguishes candidates already admitted. A two-candidate
key check can preserve bounded access, but its key representation must be learned
or explicitly declared privileged. Exact supplied keys are a diagnostic ceiling,
not a result about cross-mention identity.

### First experiment, before routing changes

Freeze the current decoder, encoder, query and address outputs. Use the same
held-out examples in these conditions: oracle singleton; oracle address with one
distinct-key/different-value distractor in the same bucket; the same two records
with reversed/shuffled order; two-record exact-key verifier; supplied gold slot
with the same gathered records. The latter bypasses disambiguation and diagnoses
decoding. Include missing-key and duplicate-key/version cases with a specified
NULL/version policy. Report each seed's joint admission/selection/answer table.

The same-bucket oracle intervention must preserve the distractor and forbid
gold-slot leakage through recency, order, address bits, masks or metadata. Failure
at this step supports a readout/selection bottleneck; it does not by itself prove
key verification is the only repair. Keep this diagnostic separate from a
matched retraining experiment comparing the original and verified readouts.

## 2. Credit assignment remains a hypothesis

The unsupervised-minus-baseline difference is only 2.18 percentage points; the
end-to-end-minus-baseline difference is 0.94 points. Report all three seed scores,
paired differences and example-level counts before interpreting either. Three
seeds provide weak uncertainty resolution, especially if examples/worlds repeat.

Once disambiguation is competent, keep the address codebook, representations,
readout, data order and update budget matched. Compare 100%, 10%, 1%, and 0%
address-label exposure using nested, fixed random masks. Do not call two sparse
rates a measured collapse threshold: they only bracket where further study is
needed. Separately warm-start with supervised addresses, remove the auxiliary
loss, and continue joint training with optimizer/weight decay unchanged. Freezing
the router would make retention trivial. Evaluate routing drift and answers both
immediately after removal and after a prespecified continuation budget.

Evaluate the same saved weights through soft and hard paths, with candidate sets,
NULL handling, temperatures and value decoder documented. Different budgets must
be counted; a dense soft training path is not a deployable bounded-index baseline.
A soft-success/hard-failure gap is evidence for execution mismatch, not proof of
its unique cause. Supervised-code targets should also be held fixed/permuted in
matched controls so target geometry is not confused with additional gradient.

For V4's conditional binary surrogate, `(L1-L0) dp/dtheta` is the derivative only
with the two losses held fixed. The full derivative of an exact binary expected
loss also has `(1-p) dL0/dtheta + p dL1/dtheta`. The present method optimizes the
sampled base value loss plus a detached local policy term, and is already
documented as not an unbiased full multi-decision estimator. Moreover overriding
an entire action tensor freezes its other coordinates. Common Gumbel noise alone
does not imply every downstream decision is recomputed. State this distinction
when comparing the ladder's training paths.

## 3. Byte reuse and mixture arithmetic

If the two splits exhaust the same token-weighted evaluation, then
`-.0069 = f*(-.0145) + (1-f)*.0211`, so `f = .78651685`: about **78.65% seen**.
Different weighting, datasets or rounded summaries can break this reconstruction;
publish the denominators. A seen-byte split is frequency-sensitive and does not
establish semantic memory, but does not by itself prove the absence of semantics.

For a fixed base model p and cache q,
`m(y) = (1-lambda)*p(y) + lambda*q(y)` and
`delta_bpb(y) = -log2((1-lambda) + lambda*q(y)/p(y))`.
If `q(y)=0` and `0<=lambda<1`, the penalty is exactly `-log2(1-lambda)`.
A penalty of .0211 would correspond to lambda about .014519 **only under those
conditions**. This gives a direct check of the proposed leakage mechanism;
without the mixture code/weight and matched base logits, it is not established.

For k in {3,4,6}, define prior matches by continuation positions j<t satisfying
`x[j-k:j] = x[t-k:t]` within the specified causal window. Fix a selection rule
(e.g. most recent j) before seeing the target. Report no-match, match/copy-correct
(`x[j]=x[t]`) and match/copy-wrong subsets, their sizes, baseline/mixture bpb and
per-seed deltas. Copy-correct is an evaluator label; it must not leak into the
gate or retrieval rule. The audit provides a prefix-only matching helper.

Compare base, fixed gate and learned causal gate with matched decoder weights
first, then separately assess joint retraining. Gate features may include prefix
match counts, confidence and disagreement; never target equality. Test lambda=0,
lambda=1 where legal, empty cache, duplicate continuations, and normalization.
Inference costs include lookup even when a gate later discards its result; retain
the user's 12:1 preference only after quality gates.

## 4. Formal verification scope and non-vacuity

The available `Hit` type carries causality and address proofs. Several lemmas
merely project these fields and do not use lookup membership. That is valid
proof-carrying construction, but it does not prove successful retrieval,
completeness, membership in original history, or correspondence to Python. An
always-empty lookup can satisfy safety and cap invariants. Add positive witnesses
and completeness properties, not only rejected mutants.

History being newest-first is a comment/precondition, not a proved ordering
invariant of arbitrary `List Occurrence`. Formal refinement should state ordering,
uniqueness, namespace isolation and the mapping from Python postings. Rank division
reconstruction holds even at cap=0 in natural-number arithmetic; actual slot
validity requires cap>0. Finite B-bit coverage also needs rank < C*2^B. These are
distinct obligations from simple reconstruction and candidate-budget bounds.

New review additions: positive paging/current-boundary/out-of-range witnesses;
positive-cap slot bound; an exact finite-mass mixture normalization theorem and
positive denominator lemma; and mutations for strict causality, page offset and
mixture weight. The mixture is a **new rational-mass contract**, not a proof of
the unavailable byte-mixture executable or floating-point arithmetic. Each mutant
must fail after an unmodified baseline passes; dependency/syntax/startup failures
do not count. The harness audits theorem axioms and rejects incomplete or extra
unapproved dependencies. It uses kernel `decide`, not `native_decide`.

Local Lean 4.15.0 again fails at startup (`failed to locate application`). Thus
these proof additions and mutation checks remain **unverified candidates**.
The user's claimed earlier successful causality mutation cannot be confirmed
without its source, build log, exact mutation and axiom output. Even a confirmed
mutation only establishes sensitivity to that mutation, not non-vacuity of every
invariant or verification of the neural implementation. Official validation guide:
https://lean-lang.org/doc/reference/latest/ValidatingProofs/

## Executed checks and publication

The standard-library audit passed 630 operational paging checks, causal 3/4/6-gram
fixtures, exact rational mixture normalization and a wrong-weight negative check.
A fully balanced four-case same-bucket fixture gives 50% for blind fixed-slot
selection and 100% for exact supplied-key verification. This is a mechanical
counterexample using the actual index, **not a rerun of the 62.11% trained model**.
No learning fit, checkpoint evaluation or new scientific accuracy claim was made.

The review, audit sources and logs are locally committed on a separate branch.
Publication of the new formal-audit tree/workflow was rejected by automatic
approval review because it considered the previous authorization limited to the
earlier LRC-002/V3/V4 payload. No alternate publication route was attempted.
Remote Lean CI therefore has not run. Publishing this new review/verification
package needs explicit authorization; the previous draft PRs remain unchanged.
