# LRR-004 — Causal role-offset locality and explicit lexical abstention

**Pre-registration — 2026-11-05.** Freeze before a new confirmatory scientific CI run. All earlier LRR-001/002/003 results, including failures, remain unchanged.

## Source-grounded diagnosis

Frozen LRR-003 checkpoints were downloaded from GitHub Actions artifact 13438891449 (evaluated head `a018f8ad787915df561aa2e1a34cf81974b0ee58`). Direct inspection of 128 reproduced H=32 test documents per seed showed the failed seed 23's hard relation-offset selector frequently chose the *fourth previous token* instead of the predicate. Gold value write gates stayed near one; the bottleneck was address construction, not capacity or missing writes.

Local exploratory **not confirmatory** fine-tunes with 160 extra updates on seed23 improved raw recall at H=32 from 32.81% (no added locality) to 51.56% (relative offset penalty 0.15). Other exploratory variants did not robustly improve. This motivated the single frozen locality weight below; none of those scores can be promoted to new evidence.

## Hypothesis and single-variable causal comparison

The source architecture remains LRR-003's 23,429-parameter local 3-layer causal convolution, learned offsets, hard four-code relation assignment and direct indexed token-position reads. For each labeled *answer-token* position (g) and question position (q), add a training-time penalty

[
L_{mathrm{local}} = \lambda\,\frac{1}{2}\sum_{u\in\{g,q\}}\sum_{\ell=0}^{3}\ell\,p_\theta(\ell\mid x_{\le u}).
]

This is a **soft proximity prior**, not an externally supplied predicate position or POS label. It has a known failure mode: longer or nonlocal dependencies can be penalized. Runtime remains unchanged and uses *no Transformer self-attention*. Training still receives answer-derived gold token positions, not just scalar rewards.

Compare:

- Control: original `answer_position_loss` at `lambda=0`.
- Intervention: **exactly the same loss/inputs/model** plus `lambda=0.15`.

The only intended variable is this offset cost. No extra relation labels, no model-parameter change, no added training updates or post-hoc seed selection.

## Frozen data and eligibility

- Reuse the same unmarked punctuation-delimited raw grammar and 4 relation families × 4 aliases, 12 names, opaque out-of-vocabulary values, and hard same-entity/wrong-relation and wrong-entity/same-relation distractors.
- New split relative to LRR-003: training **excludes ordered aliases (1,2) and (3,0)** in each family; dev uses (3,0), confirmatory test uses (1,2). All lexemes appear during training; do not call it unseen-word generalization.
- Confirmatory independent train seeds: **31,32,33,34,35**; 700 AdamW updates × 48 examples, same random documents per matched seed in both arms, learning rate 0.002, batch/model/init identical.
- Eval H={6,32,128}, **64 new documents/seed/H/arm**, fixed eval seed formula `75_000_000+seed*1_000_000+H*1000+i`. Absolutely no TEST-driven tuning.
- CPU single-thread and deterministic session where supported; record PyTorch version/threads, exact evaluated Git SHA, run ID, checkpoint digests/artifacts, training wall time and optimizer steps. No access to protected source or TEST except the declared frozen generator.

## Controls and challenges

1. The unchanged `lambda=0` model.
2. Grammar-aware symbolic union-find trained from **the same answer-labeled training documents**, not hidden family names: existing strong non-neural ceiling.
3. Original trained causal Transformer pointer only as *historical, nonqualified* baseline; do not infer attention inferiority from it or claim a new matched strong Transformer run.
4. **Out-of-support predicate**: replace only the question relation word with `unseenverb`. Compare original neural answers and a simple, explicitly engineered support guard that rejects a query predicate absent from the observed train query-word vocabulary. This is a lexical open-set guard, **not learned uncertainty**.
5. **Known-but-absent relation**: move the only matching target-family fact from the queried entity to a different entity without changing its surface predicate/answer; no correct fact remains. Measure false-answer rates (not a recall success).
6. **Intervening-adverb challenge**: insert `now` between the target fact's predicate and its value. The true predicate is now farther away; this is a *challenge-only* falsifier for locality bias and not a training distribution.
7. Query- and gold-offset histograms, write gate precision, distinct address counts, abstentions, entity-key ablation and literal-relation ablation. Persist every failed seed.

## Precommitted decision rules

- **Optimizer stability:** intervention meets H=32 recall@1 >=0.70 on at least 4/5 seeds; mean H=32 >=0.85.
- **Causal improvement:** intervention mean H=32 recall@1 improves >=10 percentage points over paired control, without any individual seed dropping >5 percentage points. Any condition unmet means improvement NOT ESTABLISHED.
- **Unknown-relation guard:** out-of-vocabulary false-answer rate <=5%, positive-case recall loss <=1pp versus same model without the guard. This can only establish a bounded engineered safeguard.
- **Known-but-absent behavior:** false-answer rate <=10% is a separate desired quality gate; a lexicon guard alone is not expected to solve this.
- **Nonlocal tradeoff:** compare challenge-only adverb success of both variants. Do not suppress a regression.
- **Scientific promotion:** none from these gates alone; a neural advantage over symbolic indexing, natural-language coreference, or attention replacement remains unestablished.

No mergers, GPU scale-up, or public model-level claims from passing a few templated tasks. LRR-005 should remove the fixed relation-code count and test genuine novel semantics with a capability-qualified attention baseline.