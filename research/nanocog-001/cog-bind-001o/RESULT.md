# COG-BIND-001O: matched word-order variation stabilized the learned route

## Result
Training with varied word order made the same routed model pass the full held rewrite in **3/3 seeds**, compared with **2/3** for familiar-order training. The diverse arm got the correct declaration, recipient, directed source and item on **12,288/12,288** combined-rewrite rows and completed **768/768** full families. The familiar-order arm got **12,034/12,288** all-stage rows and **723/768** full families. Familiar held performance stayed perfect in both arms. The improvement mainly removed a failure in one seed; it was not a large gain on an intrinsically difficult panel for every seed.

## Matched design and limits
Both arms were newly initialized with the exact same 001L routed architecture (17,370 trainable parameters) and paired initializations, optimizer schedules, role labels and auxiliary losses. The only intentional intervention was word order in TRAIN. All corresponding clauses and questions share the same token multiset and exact token count between arms, not merely equal overall corpus size. Both arms received 1,024 updates, 65,536 query exposures and 3,145,728 nonpadding token presentations per fit. The familiar curriculum repeats one order per declaration, transfer and question voice; diverse training uses order variants 0–2 across the same worlds, including all 27 registered triples.

The 512 TRAIN, 64 DEV and 256 HELD families use disjoint exact alias associations and transfer pairs. Names and item identities are fresh relative to the prior series and seen in this study's TRAIN. Each held world supports a familiar view, single declaration, transfer or question rewrite, and the combined rewrite, keeping truth and placement fixed. Exact held order 3 was absent from TRAIN in both arms and from DEV, but all atom types are familiar. These are controlled word-order permutations, not newly acquired predicates, arbitrary language, or a complete factorial test. No parser or truth enters native decision-making.

## Held measurements across three matched seeds
| View | Native exact rows, familiar / diverse | All-stage exact rows, familiar / diverse | All-stage complete families, familiar / diverse |
| --- | ---: | ---: | ---: |
| Familiar | 12,288 / 12,288 | 12,288 / 12,288 | 768 / 768 |
| Declaration only rewritten | 12,066 / 12,288 | 12,062 / 12,288 | 727 / 768 |
| Transfer only rewritten | 12,288 / 12,288 | 12,288 / 12,288 | 768 / 768 |
| Question only rewritten | 12,288 / 12,288 | 12,288 / 12,288 | 768 / 768 |
| All three rewritten | 12,038 / 12,288 | 12,034 / 12,288 | 723 / 768 |

Each row denominator is 12,288; each family denominator is 768 model–family observations over 256 underlying families, not 768 independent cases. A complete family requires all 16 presentations correct. Combined strict all-stage families by seed were 212/256, 256/256 and 255/256 for familiar training versus 256/256 each for diverse training. The fixed-seed paired family-bootstrap difference is **+5.86 points** (descriptive 95% interval **+4.30 to +7.42** on these families, not a population interval over training seeds).

Both arms found the unique correct declaration and decoded the item from the supplied correct transfer on every combined row. On the weak familiar-training seed, composed recipient was 4,068/4,096 and correct directed record 3,846/4,096. The declaration-only rewrite reproduces most of the weakness; finer stage accuracies are diagnostic and do not establish the internal cause.

## Integrity and next
Six scientific fits completed, 36 blind prediction packets covering 129,024 prompt rows, with independent raw recount. The first launch had a nested manifest-access error in both processes before model initialization or optimization. Those logs and the original source are preserved. A one-line pin accessor repair was separately reviewed before the first actual fit; no old weights or results existed and the model, data, schedules and metrics did not change.

Use the diverse route as a bounded reference for a fresh joint experiment on answerable and unanswerable inputs under wider phrasing variation and evidence arrangements. Test coverage and per-reason false accept rates on groups and templates not used in TRAIN or DEV. This run did not train or qualify an abstention gate and provides no authentic-language or inference-cost qualification.

Data pin `b7afe24589b2411e3727aef3e77d09bcf63d11cd788fbbbba01c8441936fbc88`; score `3e187285820184ff85f4827a9e6c6d08ed098cfa5604974ec3baff0f67bbc66d`.
