# CASM-Y explicit answer-state pilot — 2026-08-30

CASM-Y tests whether separating a persistent proposed-answer state `y` from latent reasoning state `z` can overcome the autoregressive degeneration seen in CASM-P and CASM-D. Gold answer bytes are masked before prompt encoding; one-step and three-step variants share exactly the same parameters, with recurrent depth reusing the same update weights.

## Run contract

- Authoritative corrected pilot run: `33283454031`
- Exact head: `4db686bfd4fff498c50a953b29be193a4d022f01`
- Training: 600 paired steps
- Variants: `answer-state-1step`, `answer-state-3step`
- Parameters: 1,556,801 for both
- Answer representation: 20 parallel persistent slots, decoded through the tied LM head
- Training targets: answer bytes + EOS
- Gold-answer leakage guard: bytes after the final `answer ` marker are replaced before encoding
- Primary metric: exact generated answer, not marginal NLL

## Internal held-out evaluation

| Metric | one-step | three-step |
|---|---:|---:|
| Easy answer NLL | 1.7163 | 1.7376 |
| Easy exact | 1/96 | 0/96 |
| Hard answer NLL | 1.7772 | 1.7591 |
| Hard exact | 0/96 | 0/96 |

For the three-step model on the hard split, recurrence does reduce marginal answer NLL:

- step 1: 1.8129
- step 2: 1.7690
- step 3: 1.7591

The likelihood refinement nevertheless does not convert to exact solutions.

## Hard six-task exact solve rate

40 examples per task, 240 total:

| Task | one-step | three-step |
|---|---:|---:|
| Associative recall | 0/40 | 1/40 |
| State tracking | 0/40 | 0/40 |
| Arithmetic | 1/40 | 1/40 |
| Rule induction | 0/40 | 0/40 |
| Graph reachability | 0/40 | 0/40 |
| Reverse | 0/40 | 0/40 |
| **Total** | **1/240** | **2/240** |

Typical unconstrained parallel outputs are malformed class mixtures such as `yos`, `nos`, `boo`, `roo`, `boom`, repeated digits, and repeated-character strings.

## Exact-balanced graph guardrail

Both variants solve 0/80 easy and 0/80 hard balanced graph examples. Neither reliably emits a valid `yes` or `no` string in unconstrained decoding.

## Recurrence-depth sweep

Using the trained three-step checkpoint on 120 hard examples:

| Effective depth | Answer NLL | Exact solves |
|---:|---:|---:|
| 1 | 1.7476 | 8/120 |
| 2 | 1.7014 | 0/120 |
| 3 | 1.6894 | 0/120 |
| 5 | 1.7006 | 8/120 |

Again, lower marginal NLL is not monotonic with solution correctness.

## Follow-up constrained whole-answer diagnostic

A separate diagnostic (`33283699678`) ranks only legal whole answers by sequence log-probability. It still reveals class-constant collapse:

- one-step graph: chooses `yes` for all 200 easy and all 200 hard balanced examples;
- three-step graph: sum-logprob chooses `no` for every example while mean-logprob chooses `yes` for every example;
- state scoring remains roughly chance, dominated by a single location (`room` or `box`) independent of the prompt.

Therefore the pilot failure is not merely a malformed renderer. The underlying answer distributions are not task-dependent enough.

## Conclusion

**CASM-Y is not promoted from the 600-step pilot.** The explicit answer state changes the failure mode but does not yet produce robust computation. The 2,000-step matched run remains the authoritative scale-up test.

In parallel, a parameter-comparable ordinary full-context causal Transformer is being trained on the same answer-only objective. That control is required to determine whether the bottleneck is CASM's compressed state representation or the tiny-model/training regime itself.
