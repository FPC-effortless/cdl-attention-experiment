# COG-BIND-TIME-PACKET-009: record-neutral clause proposals followed by grouping

**Outcome: the new design restores much of the binding behavior, but neither arm passes qualification.** The record-neutral packet model answered 2,416, 2,112 and 2,436 of 3,072 exact nested-layout cases (78.65%, 68.75% and 79.30%). The freshly fitted global locator answered zero in all three seeds. The packet model also improved exact TRAIN routing from 389–820 out of 6,144 to 5,456–5,540, but each seed still fell short of the 95% TRAIN requirement. It likewise missed the 95% novel exact-case floor. The positive paired difference is a specific comparison against a control that failed to learn this workload, not a strength or efficiency claim against a qualified baseline.

**The remaining errors are mostly in what the clauses contain, not in locating cues or in the final grouper.** The model recovered all four true cue positions in every novel row, and its raw clause scores and grouped evidence did not change across all 768 four-question sets per seed. Once aligned by the correctly predicted cue, 11,580/12,288, 11,248/12,288 and 11,604/12,288 clause packets had all five roles right. Yet only 2,416, 2,104 and 2,436 novel rows had every clause packet correct before grouping. The finite composer returned an exact answer in all but eight structurally groupable rows across the three novel evaluations, with all eight in seed 20471. Four independent role-pointer decisions at each cue can still select incorrect occurrences; validating the grouping after those selections cannot reconstruct missing evidence.

The next test should keep the now-working cue detector and compare independent role maxima against **coherent tuple selection**: let each cue nominate a few complete alternative clause packets and choose only a consistent set, with explicit equal-budget controls and exact source-position scoring. Test whether that extra search fixes the specific role-occurrence errors before modifying memory or growing the model. This suggestion is not a tested result.

## Trained results

| Seed | Arm | TRAIN exact route | Familiar exact cases | Novel exact cases | Phase pairs | Anchor flip | Anchor invariant | Exact quartets | Exact named selected evidence | Groupable packets | Exact query | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 20371 | global | 434/6,144 (7.06%) | 27/3,072 (0.88%) | 0/3,072 (0.00%) | 0/1,536 (0.00%) | 0/768 (0.00%) | 0/768 (0.00%) | 0/768 (0.00%) | 0/3,072 (0.00%) | 0/3,072 (0.00%) | 3,072/3,072 (100.00%) | FAIL |
| 20371 | packet | 5,456/6,144 (88.80%) | 2,492/3,072 (81.12%) | 2,416/3,072 (78.65%) | 1,208/1,536 (78.65%) | 610/768 (79.43%) | 598/768 (77.86%) | 604/768 (78.65%) | 2,416/3,072 (78.65%) | 2,416/3,072 (78.65%) | 3,072/3,072 (100.00%) | FAIL |
| 20471 | global | 820/6,144 (13.35%) | 35/3,072 (1.14%) | 0/3,072 (0.00%) | 0/1,536 (0.00%) | 0/768 (0.00%) | 0/768 (0.00%) | 0/768 (0.00%) | 0/3,072 (0.00%) | 0/3,072 (0.00%) | 3,072/3,072 (100.00%) | FAIL |
| 20471 | packet | 5,468/6,144 (89.00%) | 2,204/3,072 (71.74%) | 2,112/3,072 (68.75%) | 1,056/1,536 (68.75%) | 526/768 (68.49%) | 526/768 (68.49%) | 526/768 (68.49%) | 2,112/3,072 (68.75%) | 2,120/3,072 (69.01%) | 3,072/3,072 (100.00%) | FAIL |
| 20571 | global | 389/6,144 (6.33%) | 24/3,072 (0.78%) | 0/3,072 (0.00%) | 0/1,536 (0.00%) | 0/768 (0.00%) | 0/768 (0.00%) | 0/768 (0.00%) | 0/3,072 (0.00%) | 0/3,072 (0.00%) | 3,072/3,072 (100.00%) | FAIL |
| 20571 | packet | 5,540/6,144 (90.17%) | 2,376/3,072 (77.34%) | 2,436/3,072 (79.30%) | 1,218/1,536 (79.30%) | 604/768 (78.65%) | 614/768 (79.95%) | 609/768 (79.30%) | 2,436/3,072 (79.30%) | 2,436/3,072 (79.30%) | 3,072/3,072 (100.00%) | FAIL |

## Packet-stage diagnosis and paired comparison

| Seed | Novel exact cue set | Novel exact clause set before grouping | Native novel difference, packet minus global (pp) | Paired-root 95% interval (pp) |
|---|---:|---:|---:|---:|
| 20371 | 3,072/3,072 (100.00%) | 2,416/3,072 (78.65%) | +78.65 | [+72.53, +84.51] |
| 20471 | 3,072/3,072 (100.00%) | 2,104/3,072 (68.49%) | +68.75 | [+62.63, +74.87] |
| 20571 | 3,072/3,072 (100.00%) | 2,436/3,072 (79.30%) | +79.30 | [+73.44, +84.90] |

## Experimental comparison

IMPLICIT-008 showed that globally assigning each word to a preordered record failed, despite perfect identification of the question's anchor and phase. PACKET-009 compares that same global locator fitted afresh against a new clause-centric model. Both see the same unsegmented passage and question in the same per-seed order, use all 22 named position targets per row, and run the same 384 optimizer updates and 49,152 repeated training rows. They share exact initial weights for the common convolutional and query trunk within each of the three seeds. The fitted global and packet arms have 31,958 and 31,331 parameters respectively. Neither sees gold boundaries or candidate lists in prediction.

The candidate learns to nominate four cue positions from the raw passage. Each nominated cue conditions a shared pointer for the action, trigger, anchor and item. The pointer uses relative positions, but can score every observed token, so this does not establish bounded or linear-time communication. This yields four complete clause proposals, independent of record numbers and of the question; only their subsequent answer selection uses the question. For training only, the true cue positions condition those local role predictions. At inference all cue and role positions must come from the model itself. A declared, deterministic stage checks raw IDs and positions, joins clauses with matching action/item keys into exactly two records and sorts their first mentions. A second deterministic stage combines selected anchor, cue and operand order to choose the exact event and source position for the item. This is learned local grounding plus an engineered composer and grouper, not an end-to-end learned temporal algebra.

Because intermediary supervision, architecture and routing all change together, this is not a one-factor test. The same raw annotations are used differently; in particular the candidate receives gold cue conditioning during training but not prediction. Loss scales, parameter counts and training compute should not be assumed equal. The newly fitted global arm is a long-running research reference, not a strong competitor if it does not learn the training task.

## Data, counterfactuals and review

Across each held pane there are 48 fresh constructed roots and 3,072 exact cases. The familiar layouts use blocked (0011) and alternating (0101) clause mentions; the withheld pane uses nested order (0110). Each aligned familiar/novel case has the same passage words, facts, and question, but a different mention order. Within each root, before/after flips and anchor switches create strict matched tests as well as cases that must keep the answer. All local language primitives are known: 'novel' here means a withheld combination of familiar pieces, not unfamiliar natural language. The fixed grammar reference is a verifier, not an inference fallback.

Exact answers require both the right record and the correct raw item occurrence, not just the same word. Per seed, annotation and answer gates include exact training competence and independently scored strict held pairs. Relative intervals resample whole roots, not factorial rows. An order-independent packet-set match means all four five-role locations agreed before record assembly. This isolates missed clause evidence from a downstream grouping issue more directly than an end-to-end answer alone.

The blind actor reads only staged raw inputs in an isolated filesystem/PID namespace; this is not a network isolation claim. Scoring uses hash-sealed traces and independent post-actor release and recount. No native execution-time result is a matched end-to-end inference comparison. Larger-language learning, long-term agent memory and real-language transfer remain untested.

The grouping rule is also limited by its key choice: this fixture makes the two action/item pairs distinct and repeats the same key across their two mentions. Two genuinely different events can reuse an action and item in ordinary text, and one event can be described by a paraphrase. Neither false merges nor paraphrased-key splits are qualified by this experiment.
