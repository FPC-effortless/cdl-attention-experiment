# COG-MATCHED-BOUNDARY-001: neither learned selector qualified

Both selectors received identical punctuation-derived clause boundaries and chronology coordinates. Changing the score from diagonal scaled dot product to additive tanh compatibility did not produce a reliable win. All six runs fell short in calibration as well as held scoring. The best additive seed beat the dot seed in several conditions, but the other two additive seeds performed substantially worse. Treat this as a failure of both fixed training recipes, not a conclusion about the best achievable performance of either function.

The 53,958-parameter models shared parameter shapes, initial tensors per seed, 1,024 updates, batch schedules, phrase encoder, key/query/value maps, and answer decoder. Exactly one compatibility rule differed. Both are soft attention-style reads. Neither got gold keys, dates, targets or scores at inference. The small synthetic source uses disjoint two-word key combinations across partitions.

## Strict all-four histories

Counts are per 256 families. The three entries within each cell are **binding / exact date / untimestamped latest mention**. “Familiar syntax” uses unseen key combinations, not fitting data. “Novel” changes both the key combinations and the combination of known modifiers. The paired rewrite adds familiar modifiers to the same rendered families, preserving answer, key, date and support. It is diagnostic and not part of the qualification decision.

| Arm / seed | Familiar syntax | Same families, paired rewrite | Novel keys and modifier combinations |
|---|---|---|---|
| Dot 4371 | 84 / 77 / 195 | 79 / 77 / 193 | 76 / 87 / 176 |
| Dot 4471 | 156 / 176 / 226 | 164 / 162 / 222 | 142 / 164 / 213 |
| Dot 4571 | 68 / 123 / 205 | 70 / 101 / 204 | 82 / 131 / 206 |
| Additive 4371 | 181 / 229 / 224 | 184 / 232 / 209 | 195 / 214 / 216 |
| Additive 4471 | 2 / 1 / 123 | 4 / 4 / 117 | 4 / 0 / 114 |
| Additive 4571 | 23 / 4 / 146 | 17 / 0 / 136 | 19 / 0 / 126 |

No run cleared the frozen strict threshold of 231/256 in all required held strata or 116/128 in all calibration strata. In the paired rewrite, the largest absolute shift was 22 families (dot 4571 exact date: 123→101); some shifts were positive. It is not accurate to describe unfamiliar modifiers as the sole bottleneck. There is no estimate of real-language transfer.

## Separate post-hoc mechanism diagnostic

After sealing the primary report, the same frozen decoder received the value vector from the correct supporting clause, rather than the learned weighted mixture. This **used gold support only for diagnosis**; it is not an improvement to the learned selector or a qualification result. It brought accuracy to **256/256 strict families for each mode in each of the six runs**, 4,608/4,608 model–family evaluations (not independent families) on the jointly shifted novel panel. Forcing the cyclic next slot yielded no complete family in any of the 18 cells. Uniform weights yielded 74/4,608 complete model–family evaluations overall, at most 19/256 in any cell, and zero in every latest-mention cell. The original report transposed these two control labels; neither the recorded outputs nor the qualification decision changed. The diagnostic reconstructs native logits and checks them against the sealed primary wire first. This shows the learned value representations and fixed answer decoder were sufficient when given the right clause on this panel; it does not distinguish bad ranking from a useful clause being diluted by soft mixing.

Using only the model’s own top-ranked clause, without gold, changed strict scores by −25 to +19 families out of 256 across cells and did not clear the gates. Across the six fits and three modes, hard reads rescued 290 previously incomplete model–family evaluations and made 243 previously complete ones fail (net +47/4,608). Hardening alone therefore does not repair this reader. Latest has a structural recency cue in this task: the correct clause is always in absolute slot 1 or 2 (zero-indexed), whereas binding and dated lookup allow all three slots.

**Decision:** Keep dot product as the simpler reference. Next, compare equal supervision for support selection versus answer-only; continue to report support rank and both soft and top-one behavior, and require a matched fixed-boundary control on fresh cohorts. Do not scale yet or start the authentic TimeQA/learned-boundary stages until the basic support read is reliable. No claims about production latency, real language or protected tests.
