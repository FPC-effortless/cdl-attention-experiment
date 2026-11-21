# COG-ATTN 091–100 functional substitution: protocol and honesty checklist

This document describes already-executed 091–096 synthetic pilots and future gates for 097–100. **The executed experiment designs were not prospectively registered before their first local runs; nothing here should be described as preregistered.** A single-action overlap was identified and removed before the final scores. All final language reports use disjoint utterance strings and disjoint ordered action pairs.

## Five disjoint functions to recover

1. Retroactive re-querying without lost raw evidence.
2. Gradient credit from final errors into learned perception and subspace encoding, while authoritative gates remain hard.
3. Continuous competing semantic hypotheses and smooth interpolation for near-neighbor unfamiliar inflections.
4. Parallel subject/role/negation/temporal specialist composition rather than sequential Python branching.
5. In-context induction of temporary rules and skills from witnessed examples, without writing into a language output head.

## Isolation logic

- The 091 test uses exact anchors, no semantic learning; compare latest-only storage to raw scan and index.
- The 092–094 test uses a fixed six-action inventory and 5 synthetic templates; full encoder-gradient vs frozen-input ablation, orderless and attention-based baselines. Test split by ordered pairs and full sentences; held inflections share stems by design.
- The 095 test uses additional supervised two-hot ambiguity targets. Compare ordinary action logits to independent candidate hypothesis logits. The ability to defer does not imply retention of alternatives.
- The 096 test is a finite program-consistency learner using an explicit 5-primitive DSL. Report accepted precision, coverage, ambiguity, contradiction abstentions and demonstration-cache control separately.
- The tiny self-attention model is a **diagnostic from-scratch baseline** only; no claim against pretrained Transformer language competence, no equal-parameter or equal-FLOP comparison, no full system timing.

## Integrity checklist

- [x] Train/dev/test sentence-string disjointness asserted.
- [x] Ordered action pair train/dev/test disjointness asserted.
- [x] Three primary and three fresh replication seeds for semantic arms.
- [x] CPU elapsed and train time recorded separately.
- [x] Original inefficient macros remain negative evidence from prior work.
- [x] Report gate errors rather than treating approval as semantics.
- [x] Unit tests cover fail-closed retrieval collision and contradicted programs.
- [ ] Independent researcher or GitHub-hosted new-machine replay.
- [ ] Authentic unseen-language dataset evaluation.
- [ ] Real COG issue-workflow/puzzle controller integration.
- [ ] Candidate-generator credit assignment or bounded semantic retrieval under collisions.
- [ ] Equal-competence, equal-cost budget attention comparison.
- [ ] Learned rule acquisition without a preinstalled DSL/inventory.
- [ ] Stable long-history benchmark at 32,768+ meaningful observations.

## Result claims and falsifiers

Claims are confined to **synthetic diagnostic mechanisms**. Strict future promotion requires holding all gates at once; greater inference speed with lower competence is rejected. Explicitly prohibit publishing a successful *agent* claim based on a reference solver, template interpreter, lexical shortcut, supplied action inventory or evaluator-specified skill.
