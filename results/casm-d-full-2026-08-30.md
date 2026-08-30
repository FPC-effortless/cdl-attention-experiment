# CASM-D full 2,000-step result — 2026-08-30

CASM-D tested whether direct answer-byte supervision through the model's own shared LM head at intermediate recurrent steps would turn recurrence into useful computation. The treatment added no parameters: both variants deploy 1,347,361 parameters and use three recurrent reasoning steps.

## Run contract

- GitHub Actions run: `33282020731`
- Head: `e0a1cbba898b9c324b7c7ed0bcb1838fb584eea3`
- Steps: 2,000 paired training steps
- Variants: `final-only-3step`, `deep-answer-3step`
- Primary endpoint: true autoregressive exact solve rate on six hard process tasks
- Graph guardrail: exactly balanced yes/no generation
- No extra training or deployment parameters in the treatment

## Training/evaluation summary

| Metric | final-only-3step | deep-answer-3step |
|---|---:|---:|
| Parameters | 1,347,361 | 1,347,361 |
| Easy LM loss | 0.3763 | 0.3966 |
| Easy answer NLL | 1.3412 | 1.3088 |
| Easy answer-byte accuracy | 47.77% | 46.41% |
| Hard LM loss | 0.5064 | 0.5179 |
| Hard answer NLL | 1.5160 | **1.4395** |
| Hard answer-byte accuracy | 41.95% | **44.72%** |

Deep supervision therefore improved answer likelihood on the hard distribution, but that gain did not translate into a large solve-rate improvement.

## True autoregressive hard-task solves

Each task contains 60 held-out hard examples (360 total).

| Task | final-only | deep-answer |
|---|---:|---:|
| Associative recall | 0/60 | 2/60 |
| State tracking | 6/60 | 7/60 |
| Arithmetic | 0/60 | 0/60 |
| Rule induction | 2/60 | 2/60 |
| Graph reachability | 28/60 | 28/60 |
| Reverse | 0/60 | 0/60 |
| **Total** | **36/360 (10.0%)** | **39/360 (10.83%)** |

The treatment gains only three exact solves out of 360.

## Exact-balanced graph guardrail

Both models collapse to the `no` class on both easy and hard balanced graph sets:

- easy: 50/100 overall = 0/50 yes, 50/50 no;
- hard: 50/100 overall = 0/50 yes, 50/50 no.

The apparently moderate graph accuracy in naturally sampled evaluation is therefore not evidence of learned reachability.

## Long-horizon generation

Deep supervision is mixed rather than systematically better. Examples from 30-example groups:

- state-12: 3/30 -> 5/30;
- state-24: 2/30 -> 0/30;
- state-48: 1/30 -> 2/30;
- state-96: 3/30 -> 6/30;
- associative-12/24/48: 0/30 for both;
- associative-96: 1/30 -> 0/30.

## Conclusion

**CASM-D is not promoted.** Direct LM-head deep supervision shapes intermediate/final answer likelihood, but the full 2,000-step run provides only a +3/360 exact-solve change, preserves balanced-graph class collapse, and does not produce robust long-horizon capability.

This supports a narrower conclusion: recurrent latent states can be made more answer-predictive, but answer-predictive latent geometry alone is insufficient. The next architecture should change the solution representation/computation itself rather than add another auxiliary or intermediate loss to the same autoregressive decoder.
