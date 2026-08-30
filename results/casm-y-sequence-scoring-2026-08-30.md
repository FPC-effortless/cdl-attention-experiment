# CASM-Y constrained whole-answer scoring diagnostic — 2026-08-30

This diagnostic tests whether the 600-step CASM-Y pilot contains the correct semantic decision even when its unconstrained parallel answer renderer emits malformed mixtures such as `yos`, `nos`, `boo`, or `room`-like hybrids.

The trained one-step and three-step checkpoints are unchanged. Instead of taking each answer slot's argmax independently, the evaluator ranks only legal complete answers by their accumulated or mean slot log-probability.

## Run contract

- Pilot source run: `33283454031`
- Diagnostic run: `33283699678`
- Graph legal answers: `yes`, `no`
- State legal answers: `bag`, `box`, `desk`, `room`, `shelf`, `tray`
- 200 examples per split
- Graph sets are exactly balanced 100 `yes` / 100 `no`

## One-step CASM-Y

### Graph

Both sum-logprob and mean-logprob choose `yes` for every example:

| Split | Accuracy | `yes` accuracy | `no` accuracy |
|---|---:|---:|---:|
| Easy | 50% | 100% | 0% |
| Hard | 50% | 100% | 0% |

The mean top-two margins are substantial (~0.384 easy, ~0.338 hard), so this is not merely a near-tie caused by the unconstrained renderer.

### State tracking

Whole-answer ranking remains near class-frequency chance:

- easy: 16.0% sum-logprob / 16.5% mean-logprob;
- hard: 16.5% sum-logprob / 16.0% mean-logprob.

Mean-logprob selects `room` almost universally: 33/34 easy `room` examples and 31/33 hard `room` examples are correct, while essentially every other class is zero. Sum-logprob is also overwhelmingly `room` with only a handful of `box` selections.

## Three-step CASM-Y

### Graph

The score aggregation method changes which constant class wins, not whether the model uses the input:

- sum-logprob chooses `no` for every easy and hard graph example;
- mean-logprob chooses `yes` for every easy and hard graph example.

Both therefore remain exactly 50% on the balanced sets with one class at 100% and the other at 0%.

### State tracking

Again the model selects one class independent of the example:

- sum-logprob: 18.0% easy / 13.5% hard by selecting `box` for essentially every example;
- mean-logprob: 17.0% easy / 16.5% hard by selecting `room` for essentially every example.

## Conclusion

**The CASM-Y 600-step failure is not just an invalid parallel-byte rendering problem.** Restricting the decoder to valid complete answers does not reveal a hidden task-dependent decision signal. The answer distributions themselves are dominated by class-constant modes.

Therefore a renderer-only repair is not promoted. The next control is a parameter-comparable ordinary full-context causal Transformer trained on the same answer-only objective. That control determines whether CASM's compressed state representation is the bottleneck or whether the current tiny-model/training regime is itself below the benchmark's learning threshold.
