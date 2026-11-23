# COG-BIND-001X: frozen event selection with competing-noun readout

The role-sensitive reader did not clear the isolated reader gates across all three seeds. It did not establish the prespecified architectural advantage across all seeds. The complete native path did not qualify.

## Experiment
Both item readers see only the event's raw continuation and predict the item associated with the fixed alpha role. Two genuine nouns compete in each event. Both reader scores exist before any event is chosen. Native scoring requires a correct event and exact raw item position. The scorer-only correct-event result diagnoses item selection; no correct event, role location, noun shortlist or typed payload is passed to the ordinary predictor. The exact three 001W phase selectors use unchanged W-format clauses/questions with no access to the new continuation. This is a controlled two-view experiment, not a test of routing on the expanded text or arbitrary role questions.

Both readers use the same 14,105 trainable parameters, raw continuation inputs, initialization and shuffled schedule within each seed, selected-event training labels and exact item supervision. The difference is averaging the token compatibility in the direct reader versus marginalizing a learned latent anchor and a linked token distribution in the role reader. 384 AdamW updates, 49,152 item targets, 1,327,104 nonpadding continuation token presentations, and 32 additional no-update TRAIN audit batches per fit; no auxiliary role labels. These arms are sample/parameter matched, not FLOP matched.

Two renderings cross the event and reader streams in four views of the same 128 held semantic roots; 2,048 rows, 1,024 before/after pairs and 512 phase-by-noun-swap quartets per seed and view. Novel means combinations of known local grammatical forms, not unfamiliar natural language. A same-bag noun swap changes which noun is assigned to alpha without changing route text. The first draft was rejected before training when nearest-to-alpha solved the intended novel views. On the frozen replacement, nearest-to-alpha, first-noun and last-noun are each 50% per novel reader renderer; the finite exact grammar witness is 100%, a ceiling, not learned performance.

## Held scores

Each cell below is a count of correct complete phase pairs; the two answers must both be exact. No pooling of seeds or views as independent samples.

| View | Seed | Correct-event direct | Correct-event role | Native direct | Native role | Frozen route ceiling |
|---|---:|---:|---:|---:|---:|---:|
| held-familiar | 20371 | 944/1,024 | 1,016/1,024 | 944/1,024 | 1,016/1,024 | 1,024/1,024 |
| held-familiar | 20471 | 1,024/1,024 | 972/1,024 | 1,024/1,024 | 972/1,024 | 1,024/1,024 |
| held-familiar | 20571 | 1,004/1,024 | 1,024/1,024 | 796/1,024 | 816/1,024 | 816/1,024 |
| held-route-only | 20371 | 944/1,024 | 1,016/1,024 | 944/1,024 | 1,016/1,024 | 1,024/1,024 |
| held-route-only | 20471 | 1,024/1,024 | 972/1,024 | 1,024/1,024 | 972/1,024 | 1,024/1,024 |
| held-route-only | 20571 | 1,004/1,024 | 1,024/1,024 | 892/1,024 | 912/1,024 | 912/1,024 |
| held-reader-only | 20371 | 0/1,024 | 28/1,024 | 0/1,024 | 28/1,024 | 1,024/1,024 |
| held-reader-only | 20471 | 60/1,024 | 0/1,024 | 60/1,024 | 0/1,024 | 1,024/1,024 |
| held-reader-only | 20571 | 0/1,024 | 0/1,024 | 0/1,024 | 0/1,024 | 816/1,024 |
| held-novel | 20371 | 0/1,024 | 28/1,024 | 0/1,024 | 28/1,024 | 1,024/1,024 |
| held-novel | 20471 | 60/1,024 | 0/1,024 | 60/1,024 | 0/1,024 | 1,024/1,024 |
| held-novel | 20571 | 0/1,024 | 0/1,024 | 0/1,024 | 0/1,024 | 912/1,024 |

## Both-novel strict gates

| Seed | Correct-event role rows / pairs / quartets | Native role rows / pairs / quartets | Frozen route ceiling rows / pairs / quartets | Correct-event phase-pair gain (root bootstrap 95% interval) |
|---|---|---|---|---|
| 20371 | 840/2,048 · 28/1,024 · 0/512 | 840/2,048 · 28/1,024 · 0/512 | 2,048/2,048 · 1,024/1,024 · 512/512 | +2.73 pp [+0.78, +4.69]; wins 28, losses 0 |
| 20471 | 284/2,048 · 0/1,024 · 0/512 | 284/2,048 · 0/1,024 · 0/512 | 2,048/2,048 · 1,024/1,024 · 512/512 | -5.86 pp [-8.98, -3.12]; wins 0, losses 60 |
| 20571 | 316/2,048 · 0/1,024 · 0/512 | 264/2,048 · 0/1,024 · 0/512 | 1,840/2,048 · 912/1,024 · 456/512 | +0.00 pp [+0.00, +0.00]; wins 0, losses 0 |

Each seed needs 95% exact rows, 90% strict phase pairs and 85% quartets, with at most a 2-point familiar row regression against direct. The role-specific gain claim additionally needs at least +5 points for correct-event phase pairs in every seed. Native thresholds stay unchanged even where inherited routing makes a gate impossible.

## Fit adequacy and scope

| Seed | Direct TRAIN exact rows | Role TRAIN exact rows |
|---|---:|---:|
| 20371 | 4,096/4,096 | 4,096/4,096 |
| 20471 | 4,096/4,096 | 4,096/4,096 |
| 20571 | 4,096/4,096 | 4,096/4,096 |

All six native TrainLab fits and six blind prediction records were verified; all predictor paths used raw actors with zero optimizer updates. The fixed W route scores and original item scores were checked bitwise across reader arms and same-input interventions. Forge split checks and an independent model-free recount bind the sealed predicted rows and paired counts. The isolated item score diagnoses this task only. No matched-quality serving advantage, unrestricted semantic competence, or general-purpose attention replacement is established.

Frozen data digest: `862e729bee3d858f00d90f2bd4ae55542f1f74b7e2bde8e1713e20650ea3f28e`. The private replay contains the exact source, data, predictions, training receipts and closure records; no private artifacts are published here.

## Decision
All six readers reached 4,096/4,096 TRAIN rows and 2,048/2,048 paired TRAIN queries, so failure to fit this specific small training slice is not an explanation. With only the reader wording changed, outcomes are identical to the both-novel correct-event outcomes; the route-only change leaves correct-event scores identical to familiar. Direct versus role complete pair counts on the novel reader are 0→28, 60→0 and 0→0. Two seeds retained perfect event routing on both novel views but failed the item gate even with the correct event supplied. A stronger frozen route by itself cannot account for or repair this failure.

Post-hoc counts on seed 20571 show that the role-compositional arm found the target on 316/512 first-position targets in one renderer and 0/512 on each of the remaining three renderer/target-position cells. These are descriptive failure strata, not additional qualification tests. Do not scale or promote this latent link. A decisive follow-up is to compare the current joint reader against the same learned extraction head applied independently to each visible subclause, with no supplied role spans, and then withhold *both* role placements and genuinely new wording under a matched curriculum. The exact grammar witness remains a diagnostic ceiling.
