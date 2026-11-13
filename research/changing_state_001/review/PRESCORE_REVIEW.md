# Independent pre-scoring review — Changing State 001

Fresh review session `/root/review_changing_state`, 2026-11-13.
Scope: benchmark implementation and frozen-checkpoint development diagnostic.
No training, neural evaluation, head-role submission, scientific admission or
resource expansion was performed by this review.

**Verdict: PASS for the declared bounded diagnostic. No blocking issue found.**
This verdict does not establish a learned architectural advantage, source runtime
parity, production cost, deletion skill or native restore/ownership correctness.

## Executed evidence

- Repository instructions and scope read; ten author unit tests rerun and passed.
- Separately authored `independent_check.py` exhaustively checked 50,625 four-event
  traces over two keys/two values, including SET, COPY, LINK, DELETE and NOP.
  All 607,500 prefix/query comparisons agreed between the map implementation and
  backward-history interpreter, including an absent third key. Queries did not
  mutate state.
- Independently parsed all 2,560 generated token streams and checked final answer,
  immediate predecessor, changed/untouched status, token boundaries, destination
  targets and admission timing.
- Checked all ten panels' exact eight-value balance and eight base blocks.
- Same independent checker passed with normal Python and `python -O`.
- Frozen local source/data hashes checked after prepare; review hashes are recorded
  in `prescore_checks.json`. The checker has no neural imports or fitting code.

## Source inspection findings

Learned CSM and historical Transformer forward calls receive only input token IDs.
Only the explicitly supplied-controller diagnostic receives destination/edit
metadata. Metadata remains available to instrumentation and scoring, which is
appropriate for these diagnostics. The supplied arm is not described as learned
routing. Original source/dependency and checkpoint hashes are verified, inference
mode is used, and both state-dict tensors and checkpoint bytes are checked for
change. This is an honest instrumented replay receipt, not a security-grade runtime
attestation or complete session-cost benchmark.

Snapshot COPY and live LINK semantics are distinct and consistent: unresolved
copies become independent absence; dangling links can recover when targets are
recreated; cycles return UNKNOWN; self-copy resolves before mutation. These are
specified task semantics and do not claim trained-model support.

## Reporting cautions and limits

1. There are only **eight base blocks**, each with eight correlated value shifts.
   Four queries, shifts, regimes and depths are not independent replications.
   The two regimes at depth one are **identical paired histories**. Do not count
   them as separate evidence or pool query counts into confidence intervals.
2. Transformer panels beyond 128 tokens are **excluded by this evaluator's scope**.
   The source itself implements longer sliding-window evaluation; exclusion is not
   an assertion of architectural incapacity or a failed score. The historical
   answer-only Transformer is not an equally supervised competent comparator.
3. The stress generator starts with four distinct values. Its depth-one panel is
   consequently a new development distribution, not an exact IID replication of
   the old training/test distributions. TRAIN complete-history exclusion does not
   establish semantic novelty, private TEST separation or natural-language transfer.
4. The descriptor `all_changed` means cycling updates across identities: at depth
   one only one identity has changed. Metrics correctly use actual changed flags.
5. The runner's filesystem hashes and module-origin receipt are useful reproducibility
   evidence but do not implement the separate memory governance/native attestation
   gate. Lifecycle methods are benchmark primitives, not untrusted-input APIs.
6. The history oracle is recursive and its exhaustive check is bounded. No claim
   of unbounded stack-safe execution follows.

## Next experiment recommendation

A conventional learned slot store using the same token frontend and equivalent
entity/destination, admission and overwrite supervision is a more discriminating
next comparison than immediately introducing new operations. It asks whether the
successful training information is sufficient without CSM's operator and link
machinery. Preserve all three seeds, original TRAIN exposure and example schedule,
record actual parameter/compute differences, and freeze the baseline before
scoring. Do not add unused parameters merely to match count. Match compatible
frontend initialization if desired, but do not call unlike full models identically
initialized. It remains an exploratory equal-information control until tuning,
competence, held-out design and end-to-end cost are qualified; reuse of these
public development panels cannot become a fresh confirmatory test.
