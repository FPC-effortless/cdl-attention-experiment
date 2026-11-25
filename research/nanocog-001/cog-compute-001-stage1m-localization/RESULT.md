# COG-COMPUTE-001 Stage 1m: Frozen routing and subject binding

**Routing caused many subject-binding failures, but not all of them.** At frozen shared ×4 checkpoints, moving half the competing fact's weight to the correct fact improved exact subject-change answers from **415/576 to 481/576**. Two training seeds reached **192/192** each under this diagnostic; the weakest improved only from **79/192 to 97/192**. The gains at native ×4 were often also attainable by moving weight away from the rival to another window. This supports testing learned subject selection, not claiming that the scorer is the only bottleneck.

## Design

Three original shared ×1 and three original shared ×4 checkpoints, no training. Reuse the exposed 768 primary and 384 value-swap prompts from Stage 1m; no new qualifications. At the same fixed weights, compare ×1 and ×4 across the entire causal autoregressive decode, using original readout and full vocabulary. For allocation diagnostics, an actor blind to the labels enumerates all ordered donor/destination pairs of the ten causal spans. Move half of the donor's weight at the native scale while holding all other coefficients and every payload unchanged. Only intervene in free generation when the original answer marker was naturally emitted. Score the exact answer and EOS predicate using the genuine next-token and native suffix logits, with no output mask. Detached scoring identifies matching and competing intact trigrams and checks exact parent parity. There are eight other valid destination spans for the same donor and transfer dose; these can overlap fact fragments and are not guaranteed empty of evidence. Opposite directions use different donors and need not have the same dose.

## Fixed endpoint crossover

| Training scale | Exact changed answers at eval ×1 → ×4 / 1,152 | Whole changed families at eval ×1 → ×4 / 288 | Whole eight-case swap families / 288 |
|---|---:|---:|---:|
| ×1 | 600 → 653 | 78 → 54 | 2 → 21 |
| ×4 | 469 → 988 | 29 → 221 | 2 → 194 |

Turning up the scale after training at ×1 did not reproduce the family gain. The strongest result required the routing rule and weights learned under that rule. This distinguishes effect at fixed weights from effect under a matched training regime. It does not establish how much of the learned effect is scorer versus downstream accommodation.

## Directed allocation at the training scale

| Outcome | ×1-trained, native → rival-to-target | ×4-trained, native → rival-to-target |
|---|---:|---:|
| Exact changed answers / 1,152 | 600 → 1,046 | 988 → 1,056 |
| Complete changed families / 288 | 78 → 230 | 221 → 247 |
| Complete eight-case swaps / 288 | 2 → 204 | 194 → 238 |

Relative to eight same-dose, same-donor transfers to non-intact spans, rival-to-target produced 1,579 wins / 48 losses across 9,216 paired decisions in ×1-trained models; 123 / 23 in ×4-trained models. These are repeated interventions, not independent cases. The ×4 arm's non-intact transfers themselves answered 8,348/9,216 correctly (90.58%), versus 91.67% for target-directed transfer. The advantage of assigning weight specifically to the target is therefore smaller than the raw lift from suppressing the rival. Direction matters: in the ×4 subject-change cases, target-to-rival transfer reduced exact primary answers to 108/576.

At ×4, primary subject families improved 80→104/144; both exact assignments improved 298→410/576, and whole eight-case subject swaps improved 54→96/144. The two strong seeds completed 48/48 whole subject swaps each under the favorable intervention; the weakest seed stayed at 0/48. The intervention is finite and does not supply exact relational semantics, so the residual cannot distinguish a wider routing defect from a representation/readout defect on its own.

**Next:** preserve the two strong seeds as evidence that better subject routing could help. On the weak seed, use a matched α/recipient probe and payload intervention before committing to a new training objective. Do not add memory, retrain by default or promote ×4.

## Custody

Six original CPU ACTs once, 17.810770 summed role seconds. No checkpoint optimization or new holdout use. WIRE and original parent checked; exact native free/marker parity and mean-weight agreement within 1.2e-7. Original detached scorer stopped solely when serializing a NumPy scalar, leaving a partial artifact preserved as failed. Isolated serializer-only recovery wrote a new file, with exact original arithmetic unchanged; independent raw recount agrees on the entire count tree. No model rerun. LOCK d6457be8fb25b94d4dcee489444d36557439b484e5370490b2982ef75d65dc1f; WIRE eae0b9cd08fbf2c0dcb27430982e773a85f5fae3371104ad4338ee44ef0c8b73; recovered score 83a73f3521f499400424bdfa7d7ce073a9c8fceb8a3259aef657d070453b699e; independent recount 9ac72c613cdab2b3f2b5acf0dc34010889e1364e4da0483d952ab37a87b151c9.
