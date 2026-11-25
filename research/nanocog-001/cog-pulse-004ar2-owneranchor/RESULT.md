# COG-PULSE-004A-R2: owner-anchored matching

**An owner-only input improved some source decisions, but did not reliably identify the full set of matching records.** The fitted owner-only metric ranked the true owner ahead of an identical-value/time rival **456/576** times, close to the equally supervised mixed-record metric (**458/576**). The owner-only representation passed exact input-invariance checks, but no evaluation panel passed qualification.

## Comparison

Three frozen R2 readers and their existing tied 48-token answer heads were used. The new adapter fit on 1,536 constructed examples containing 12 owner tokens. Its 768 separate open examples used seven different owner tokens in both queries and records. All seven had been seen in the earlier frozen reader's training: the holdout applies to the new adapter only, not to the complete system. The open banks and prompts are distinct from checked ancestors, but 100 of its 105 individual records and all 79 of its owner–value pairs occurred somewhere in older studies.

The mixed-record match branch froze time to a constant but retained the real value. The owner-only branch used the same frozen record encoder, but fixed both time and value in the grammar-defined input copy. The real query kept its original role-specific context. Query and record therefore use different token contexts; same-owner matching is not a symbolic equality check or guaranteed by identical vectors. All fitted arms used the same labeled training rows, the same bank and per-slot loss, the same update count and a fixed threshold. The two learned metrics had identical capacity. The owner-only cosine comparator used two fitted coefficients. No open owner labels or gold slots entered the features. The frozen router and answer head received the original records, not the copies.

The open panel balanced repeated-owner (K2) and three-distinct-owner (K3) banks. Each composition had 12 roots, 192 hit/no-match pairs and 96 true-record vs same-value, same-time foreign-record comparisons. It included 16 factual owner transposition pairs per composition where two owner tokens exchange positions, changing the source slot but not the answer; absence controls and target/rival value edits were separate. This was not a time-swap experiment.

## Source and answer results

Counts are in seed order 173 / 271 / 907.

| Metric | Composition | Mixed learned metric | Owner-only cosine | Owner-only learned metric |
|---|---|---:|---:|---:|
| True record ranked over same-value/time rival /96 | K2 | 82/78/74 | 88/64/52 | 84/76/64 |
| True record ranked over same-value/time rival /96 | K3 | 80/80/64 | 92/68/60 | 80/84/68 |
| Exact shortlist on both hit/no-match cases /192 | K2 | 26/24/8 | 8/0/0 | 8/48/32 |
| Exact shortlist on both hit/no-match cases /192 | K3 | 12/20/4 | 16/0/0 | 8/8/0 |
| Exact shortlist on both owner-transposed cases /16 | K2 | 3/5/1 | 2/0/0 | 4/10/2 |
| Exact shortlist on both owner-transposed cases /16 | K3 | 1/5/1 | 2/0/0 | 0/10/0 |
| Both final answers correct, original router /192 | K2 | 28/50/18 | 24/0/0 | 24/66/38 |
| Both final answers correct, original router /192 | K3 | 50/58/40 | 36/2/0 | 36/38/40 |

The owner-only features and scores were bit-identical under the tested target and rival value edits, as expected from the input construction. That did not ensure correct discrimination. The owner-only learned metric correctly accepted the true record and rejected its equal-value/time rival in **204/576**, up from **126/576** for the mixed metric. But it also falsely accepted more of those rivals overall (**224 vs. 118**). Complete bank gate pairs were **376/1,152** for both, while exact answer pairs using native routing were **242 vs. 244**. The fitted owner-only cosine was particularly conservative: in two seeds it accepted at most 8 of 192 true-hit cases per composition. Its rank counts are also the raw cosine ordering because its fitted scalar slope is positive.

Separately, selecting the latest visible time among learned positive slots found the correct source on **408/1,152** hit evaluations with the owner-only metric, compared with **274/1,152** for the mixed metric. It did not improve paired full answers under that selection rule (**100 vs. 116**). For an engineered ceiling, literal owner equality and chronology supplied the actual current record to the same frozen decoder. The decoder gave **140/128/128** exact factual answers for K2 and **152/144/132** for K3, out of 192 per seed. In shared-value cases a correct answer alone is not proof of source identification.

## Decision

Do not promote the matcher or retune this open panel. The typed owner input removed payload dependence by construction and improved correct chronological source selection in this diagnostic, but decisions on the true owner-only swaps remained weak and the answer pipeline did not improve. Test role-aligned owner representations next, with an explicit test that different surface forms for the same owner still match, and keep rank, complete shortlist and open-set rejection separate. A shared identical-token cosine or literal equality remains an engineered control, not a learned alias capability. The writer and larger-scale language fitting remain out of scope.

All six native CPU roles ran once; the independent evaluation matched the entire locked score. Counts span shared roots and repeated seeds, not independent samples. No protected or paid run, serving benchmark or natural-language claim is made.
