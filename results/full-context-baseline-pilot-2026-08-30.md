# Full-context answer-only Transformer pilot — 2026-08-30

This is a benchmark sanity control for the CASM research line. It removes compressed memory, persistent CASM state, and recurrent answer-state machinery, while keeping a comparable parameter budget and the same corrected synthetic process tasks.

## Run contract

- Authoritative run: `33283900059`
- Exact head: `6ac4ce1ecd0f3d048bf52f658d4aef4c8c351409`
- 600 training steps, batch size 8
- Ordinary full-context causal Transformer
- Parameters: 1,425,760
- Training objective: answer bytes plus terminal EOS only
- Generation prompt ends at the literal `answer ` marker; gold answer bytes are absent
- Primary metric: true autoregressive exact answer
- Graph guardrail: exact 50/50 `yes` / `no`

## Teacher-forced answer metrics

- Easy answer NLL: 1.3496
- Easy answer-byte/EOS accuracy: 46.03%
- Hard answer NLL: 1.5313
- Hard answer-byte/EOS accuracy: 43.37%

Training answer NLL falls from 5.606 at step 1 to around 1.0–1.3 near the end of the run, showing that the model is learning the answer distribution strongly.

## True hard-task exact solves

30 examples per task, 180 total:

| Task | Exact |
|---|---:|
| Associative recall | 0/30 |
| State tracking | 4/30 |
| Arithmetic | 0/30 |
| Rule induction | 0/30 |
| Graph reachability | 14/30 |
| Reverse | 0/30 |
| **Total** | **18/180 (10.0%)** |

State tracking is itself class-collapsed in the shown examples: the model repeatedly emits `box`, correctly solving the examples whose gold class happens to be `box`.

## Exact-balanced graph guardrail

Both easy and hard sets contain 40 `yes` and 40 `no` examples.

- Easy: 40/80 = 50%, with `yes` 0/40 and `no` 40/40.
- Hard: 40/80 = 50%, with `yes` 0/40 and `no` 40/40.

The baseline therefore predicts `no` for every balanced graph example.

## Interpretation

The 600-step full-context baseline does not establish that CASM compression is the primary bottleneck. A conventional full-context Transformer with similar capacity also learns answer likelihood/prior structure without learning the underlying algorithms.

Notably, its hard answer NLL (1.531) after only 600 steps is already close to the 2,000-step CASM-D final-only NLL (1.516), yet exact solving remains around 10%. This is further evidence that answer likelihood is not a sufficient architecture metric for this benchmark.

A 2,000-step full-context control is required before deciding whether the six-task suite is a useful discriminator at this parameter scale.
