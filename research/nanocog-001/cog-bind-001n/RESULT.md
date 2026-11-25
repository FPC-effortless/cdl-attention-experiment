# COG-BIND-001N: learned evidence gating works on familiar wording, fails under a paired rewrite

## Decision
Both learned gates passed the prospective familiar-wording screen in all three seeds. They accepted **all 1,920 valid decisions**, all correct, while falsely accepting **5/7,680** invalid decisions for the confidence-feature gate and **3/7,680** for the pair-evidence gate. The DEV-selected fixed score-gap rule falsely accepted **382/7,680** at the same valid coverage. This is a bounded win for learned rejection, not evidence that the pair features are necessary.

With the same held scenes rewritten in unfamiliar forms, both gates failed all three seeds. Valid acceptance dropped to **655/1,920 (34.1%)** for confidence features and **539/1,920 (28.1%)** for pair evidence. Every accepted valid answer was exact, but low coverage and surviving invalid answers make this unsuitable as a reliably calibrated abstention policy.

## Matched design
Three exact routed 001L backbones were frozen. They score eight clauses by the same computation generalized over the clause dimension. Gate fitting changes only new weights, not the backbone, witness choice, or answer. For each checkpoint, fit two equal 725-parameter classifiers to the same TRAIN exposures: a confidence-feature MLP and a pair-evidence MLP. Both get 12 shared rank gaps. The first adds 12 uncertainty/score features; the second adds 12 learned stage and pair measurements. No parser fields, correct witnesses, counts of correct candidates, or reference answers reach either head. The typed grammar parser is only an exact reference, not the learned intervention. The literal alias-count and DEV-selected joint-gap rule are separate controls.

TRAIN/DEV/HELD are 256/64/128 group-disjoint synthetic families (6,400/1,600/3,200 queries). The novel wording view gives the same held groups and truth, not a second independent sample. All atom types were known to the frozen parent. Context, length and lexical marginals match by reason; within a scene the query alias changes which evidence is relevant. Counting literal aliases distinguishes absent and conflicting declarations, but misses the two transfer failure types. We require ≥90% valid coverage, ≤5% false accepts in each invalid class, and every accepted valid decision to give the correct frozen source and item.

## Held evidence (three checkpoint repeats)
| Decision rule | Familiar valid accepted / 1,920 | Familiar invalid accepted / 7,680 | Rewritten valid accepted / 1,920 | Rewritten invalid accepted / 7,680 |
| --- | ---: | ---: | ---: | ---: |
| Frozen route without rejection | 1,920 | 7,680 | 1,920 | 7,680 |
| Literal alias count | 1,920 | 3,840 | 1,920 | 3,840 |
| DEV-selected joint score gap | 1,920 | 382 | 1,000 | 373 |
| Learned confidence features | 1,920 | 5 | 655 | 370 |
| Learned pair evidence | 1,920 | 3 | 539 | 422 |

The fixed rule chose strict gap >2 on DEV for all three checkpoints. On familiar wording, its false accepts were missing alias 67 and missing incoming transfer 315; both learned heads only falsely accepted missing incoming transfers (5 confidence; 3 pair). Each learned head got 383/384 whole families right by accepting correct valid cases and rejecting all invalid ones; exact reason classification across an entire family was 372/384 for confidence and 377/384 for pair. These are repeats on 128 underlying held families, not 384 independent ones.

On rewriting, subtype false accepts ordered as missing alias, conflicting alias, missing incoming transfer, conflicting incoming transfers were **0, 11, 197, 162** for confidence and **0, 60, 252, 110** for pair (1,920 per subtype). The pair head showed no transfer advantage: one seed accepted zero rewritten valid cases. The simple DEV rule retained more valid cases than either learned head but still missed the prescribed coverage and per-type risk targets.

The frozen route itself was exact on only **1,683/1,920 (87.7%)** rewritten valid rows, with **591, 453 and 639/640** by seed. Because the gate cannot fix answers, this is an upper bound on exactly answered valid coverage for any reject-only policy using this route; the 453/640 seed cannot meet the 90% exact-coverage target even with a perfect gate. The learned heads retained far less than this ceiling, so both routing and evidence gating need attention.

## Integrity and next
The six tiny fits and 18 pre-scoring gate prediction files closed before evaluation. The first scorer stopped before reading predictions because it compared the intentional group-level `syntax` labels as if they were semantic truth. A scorer-only amendment ignored that field and required exact equality of every other field; independent checks verified the expected view tags and all 3,200 truth rows were unchanged. No features, fits, output bytes, or decisions changed, and no model was rerun. The original and amended scorer are preserved alongside the independent recount.

Next: qualify the route itself under more diverse wording, then train and calibrate the rejection decision on separate wording families. Keep explicit evidence checks for absent and contradictory facts; judge valid coverage, correct selected witness and each invalid class jointly. Neither this study nor the parser establishes a general natural-language, safety, or serving-cost result.

Data pin `a6e046326afa0bf074f682bf49a626c44a402aff7be41769aa38aadb0e251ceb`; score `1fd4c16b510e153339e6176d6e5b68e7f73c6c60b857aff370fda58cadf53a49`.
