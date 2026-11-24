# COG-PULSE-004A-R2: content matching removed slot dependence, but not answer errors

The content-aware reader answered the same way after moving the records across the tested slots. It also handled more current-versus-obsolete cases than either fixed-slot comparator. **None of the three seeds met the preregistered reader gate.** Most importantly, putting the largest weight on the right record did not guarantee a correct answer.

## Complete case families on fresh open fixtures

Each cell is the number of complete **families out of 24** that were exactly correct across every case and all eight tested slot placements. Within each cell, seeds are ordered 173 / 271 / 907.

| Cohort | Fixed slots | Fixed slots + null | Content matching + null |
|---|---:|---:|---:|
| Single record | 9 / 9 / 11 | 9 / 12 / 14 | **16 / 14 / 6** |
| Current + unrelated | 0 / 0 / 0 | 0 / 0 / 0 | **11 / 8 / 4** |
| Current + obsolete | 0 / 0 / 0 | 0 / 0 / 0 | **3 / 5 / 2** |
| Current + obsolete + unrelated | 0 / 0 / 0 | 0 / 0 / 0 | **2 / 2 / 2** |
| Occupied bank, no matching record | 0 / 0 / 0 | 0 / 0 / 0 | **4 / 6 / 3** |
| Same facts, switch whose fact is asked | 1 / 0 / 0 | 0 / 0 / 0 | **9 / 10 / 8** |

The primary comparison held the learned null option fixed. On the three-record task, content matching gained **2/24** complete families in each seed, below the registered improvement margin of four and well below the absolute floor of 22. Singleton performance also fell by eight complete families on seed 907. Two content seeds missed the TRAIN foothold on the query-switch cohort, and the fixed-slot controls were not generally adequate on the open task.

## What the interventions showed

- **Tested slot invariance held.** Content matching caused no categorical change upon translating any of the 1,824 cases across eight slots, or under the 912 reciprocal record-order swaps, on any seed. Maximum full-logit drift was 7.63×10⁻⁶. With fixed slots plus null, outputs changed for 306 / 447 / 389 distinct cases under slot translation. This is evidence for invariance under these transformations, not arbitrary language or storage layouts.
- **Routing improved more than answering.** In the three-record condition, the current record had the unique largest weight in 5,280 / 5,568 / 5,056 of 6,144 placement–case evaluations; outputs were right in only 3,184 / 3,376 / 2,320. Thus **at least 2,096 / 2,192 / 2,736** had the right top-weight record but a wrong answer, a conservative lower bound from the marginal counts. With a singleton, the correct record always ranked first but only 608 / 576 / 400 of 768 answers were right. Largest weight is not the same as an uncontaminated, fully decodable read.
- **Chronology improved but stayed weak.** When only the visible ordinals of the two same-owner facts exchanged, both three-record answers were right across all placements in **118 / 130 / 64 of 384** matched pairs, versus **6 / 6 / 7** for fixed slots plus null. A literal identity-and-latest control gave all 1,824 open answers; simply choosing the globally latest record got only half the three-record cases. The learned model received no explicit rule, rank label, or address target.
- **Null did not qualify.** For occupied banks with genuinely absent query owners, the content arm gave the correct no-evidence token in 672 / 736 / 656 of 1,536 placement–case evaluations. The null option had the unique largest weight in only **0 / 304 / 0**. An explicit no-evidence output and a genuinely selected null are separate events. Disabling the read removed all complete factual families; that establishes reliance on memory here, not reliable rejection when memory is present but irrelevant.

## Scope and next test

All nine fits started fresh; same shared initialization and row schedule per seed, same ordinary 48-way answer loss and 640 updates × 48 rows per arm. Query and raw records used continuous encoder states; only the content arm compared individual record state to query state before the same payload mixture and original answer head. Both null arms had identical query-dependent, zero-payload null capacity. Allocation was 33,959 parameters for the native baseline and 34,008 for each null arm, with 128 otherwise allocated slot-anchor parameters disconnected only in the content arm. There was no separate fitting for modules, address supervision, old checkpoint, protected test, paid compute, or measured inference advantage. The fresh CAL data are constructed, not natural-language evidence.

**Next:** start with the sealed routing-versus-answer trace. Then, under a separate no-fit protocol, test the original answer head with just the correctly selected payload at full strength, at its native weight, and with the other contributions restored. That separates read dilution from a payload or answer-head limit before deciding what to change or train. A successful slot-invariant reader is not yet evidence for learned writing, broad chronology, or an attention replacement.

Original score SHA-256: `4cff37076fd9984d544fe3707ed4a6f28852274b9540627d6002050cc3402159`. WIRE: `b050a4de9b9f705bcf8d572f5e8cf237de3f986eb1b2f48973261ece8040867d`. Independent model-free recount verified every field and all 131,328 native full-vocabulary rows plus their read-disabled counterparts: `74ac221f5708efadcca2d85d8e7954b3dd170829efc3b1bf5e6feac62254b0c1`.
