# COG-PULSE-004A-R2: record-level owner supervision

**Record-level labels made the matcher more conservative, not better at distinguishing owners.** Compared with the identical bank-only fit, the new fit accepted fewer same-value, same-time records from a different owner (114 to 50 of 576 examples across seeds and compositions). But it also accepted the true current record less often in every evaluation panel. No panel qualified.

## Comparison and scope

Three frozen PULSE R2 readers and their tied full-48 answer heads were tested using 1,536 constructed training rows and 768 separate open rows. The two compositions were a repeated owner plus a different owner (K2) and three distinct owners (K3). Each open composition had 12 roots, 192 same-bank hit/no-match pairs, and 96 factual cases where the current record and a rival record shared exactly the same value and timestamp but differed in owner. Separate one-token value edits tested the queried owner's value, which changes the correct answer, and the rival's value, which does not. Neither edit changes which owners match. This study did not swap an owner's identity across worlds or intervene on timestamps.

Train and open sets had disjoint banks, prompts, owner–value edges and rosters within this experiment. Full banks and prompts were new to the checked lineage; all 89 distinct open owner–value edges occurred somewhere in older studies and 86 of 112 constituent records recurred. This tests new combinations of a familiar grammar and atoms.

All three arms fit the same training actors, applied the same fixed threshold, and used a separate clock-neutral copy for identity scores. Controls were fitted maximum cosine and the same per-record metric trained on bank existence alone. The intervention added a fixed mean binary cross-entropy term on all three training slot labels to the same bank loss, without changing capacity, features, initialization or optimizer budget. Current and obsolete records of the queried owner were both positive. No owner labels, answer tokens, or supplied slots entered open inference. The original router, answer head, record values, and times were untouched.

## Open results

Counts are in seed order 173 / 271 / 907. The first two metrics have 96 cases per seed and composition; the remaining metrics have 192.

| Metric | Composition | Cosine | Bank-only metric | Record-supervised metric |
|---|---|---:|---:|---:|
| True current accepted and indistinguishable-by-value rival rejected | K2 | 16/4/8 | 16/30/24 | 16/26/22 |
| True current accepted and indistinguishable-by-value rival rejected | K3 | 10/6/12 | 18/24/18 | 20/14/18 |
| Same-value, same-time rival falsely accepted | K2 | 30/32/28 | 26/22/8 | 10/8/4 |
| Same-value, same-time rival falsely accepted | K3 | 34/28/22 | 24/16/18 | 8/10/10 |
| Both same-bank applicability decisions correct | K2 | 72/50/46 | 88/70/48 | 74/56/48 |
| Both same-bank applicability decisions correct | K3 | 50/48/38 | 34/46/34 | 24/34/32 |
| Both full answers correct, original router and head | K2 | 66/48/24 | 62/56/32 | 60/36/34 |
| Both full answers correct, original router and head | K3 | 40/48/16 | 42/44/28 | 24/28/24 |

Correctly accepting the true record while rejecting its hard rival fell from **130 to 116 of 576**. Accepting the actual current record across all factual queries fell from **432 to 304 of 1,152**. On the 96 repeated-owner cases per seed with two valid records of different ages, accepting both fell from **32/20/8 to 20/0/4**. The exact positive set on genuine no-match rows improved from **588 to 816 of 1,152**, while it fell from **220 to 182** on genuine hits. Unchanged output decoding with the original route yielded **206 of 1,152** complete hit/absence answer pairs, versus **264** for the bank-only fit and **242** for cosine.

In a post-hoc check that ignores the decision threshold, the true current record scored above the same-value, same-time rival in **466 of 576** comparisons, versus **464** for the bank-only fit and **456** for cosine; no scores tied. This suggests the dominant change on this comparison was which records crossed the cutoff, not a large improvement in their relative ordering. It does not establish an alternative calibrated threshold or change qualification.

A separate chronological rule chose the highest visible time only among predicted positive slots and rejected tied choices. It selected the correct current record **250 of 1,152** times for record supervision, versus **286** for bank-only. On the grammar-only source ceiling, literal owner equality and chronology supplied the correct original packet to the unchanged decoder, recovering **176/172/136** factual answers for K2 and **180/164/144** for K3 (of 192 per seed). The answer head still accounts for some errors, but the large gap before decoding remains.

## Decision

Do not promote the loss or retune its cutoff on this panel. It reduced false acceptance partly by dropping true records, including earlier records for the right owner. A more direct next diagnostic is to separate a learned owner query/key representation from payload at encoding, rather than asking a small score adjustment to recover owner identity from the same mixed record features. Use a typed owner-only input only as a bounded diagnostic with matched simpler controls; it would not by itself demonstrate identity extraction from natural language. The later source selector, decoder, and writer remain separate questions.

All six bounded CPU roles ran once. Only small reader-side metrics were fitted; no protected cohort, paid work, or writer changes. An independent arithmetic evaluation reproduced the locked score. Counts share constructed roots and are repeated across seeds, not independent samples or a serving benchmark.
