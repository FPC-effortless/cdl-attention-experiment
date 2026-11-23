# GEN-002: Can the 1M decoder answer about the right subject?

The original 993,574-parameter decoder continued from each corresponding GEN-001 checkpoint. The factual statements and questions were the same length in every evaluation condition. Both arms generated freely from all 65 original characters, with no lookup, output replacement, learned answer head, or answer choices. Exact answers include the complete four-letter word and final period.

**0/2 continuations passed all preregistered gates.**

| Held result | Parent 1 | Continued 1 | Parent 2 | Continued 2 |
|---|---:|---:|---:|---:|
| familiar, exact rows | 159/512 | 233/512 | 158/512 | 252/512 |
| familiar, complete families | 0/128 | 9/128 | 0/128 | 8/128 |
| novel, exact rows | 134/512 | 212/512 | 150/512 | 211/512 |
| novel, complete families | 0/128 | 6/128 | 0/128 | 6/128 |
| reversed, exact rows | 140/512 | 154/512 | 179/512 | 177/512 |
| reversed, complete families | 1/128 | 1/128 | 0/128 | 2/128 |
| relevant, exact rows | 83/256 | 97/256 | 86/256 | 97/256 |
| relevant, complete families | 28/128 | 25/128 | 18/128 | 19/128 |
| irrelevant, exact rows | 73/256 | 102/256 | 73/256 | 104/256 |
| irrelevant, complete families | 15/128 | 24/128 | 18/128 | 29/128 |
| unsupported, exact rows | 104/256 | 103/256 | 89/256 | 95/256 |
| unsupported, complete families | 35/128 | 35/128 | 25/128 | 31/128 |

Continued 1: missed held-familiar, held-irrelevant, held-novel, held-relevant, held-reversed, held-unsupported, train.

Continued 2: missed held-familiar, held-irrelevant, held-novel, held-relevant, held-reversed, held-unsupported, train.

**The bottleneck remained binding, not syntax.** The two runs generated well-formed strings on every held row but got both queries right with exactly the same novel passage only 35/256 and 37/256 times. Neither had mastered the training cases (1,469/2,048 and 1,507/2,048 exact factual answers). When the requested subject was absent, all 153/256 and 161/256 errors copied a value from one of the two visible but unrelated claims. The equal-width result rules out the longer input as the only explanation for GEN-001’s two-fact failure, but does not identify whether model size, supervision, or training budget limits this architecture.

A complete familiar, novel, or reversed family requires both query subjects in both factual worlds. A relevant edit must change the answer; an irrelevant edit must keep the correct answer. The literal output ‘none.’ is a learned missing-subject response in this small grammar, not general abstention. This is a current-input language test, not temporal revision, persistence, or an attention comparator.
