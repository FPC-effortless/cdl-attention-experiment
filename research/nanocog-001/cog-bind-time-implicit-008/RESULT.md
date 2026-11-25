# COG-BIND-TIME-IMPLICIT-008: binding without supplied clause boundaries

**Outcome: neither label regime qualifies. Six of six runs abstained on all 3,072 held nested-layout cases.** Every fit selected the exact query anchor and phase on all those cases, but none assembled a packet that the fixed composer could accept. The failure already appeared within TRAIN: even the best run assembled and routed only 896/6,144 exactly, below the registered 95% floor. This is not a test showing that quarter supervision is as good as full: the novel comparison is at the floor and a functioning full-supervision baseline was never reached.

Across novel rows, every fit recovered at least one complete correct clause packet. Allowing arbitrary reordering recovered all four correct clauses on just 7–88/3,072 rows per run, compared with zero with the assigned slots. That diagnostic shows that some errors really were wrong slot assignments, but sorting the four existing packets would not recover most cases because complete packet content was usually missing or wrong. It does not test what a different decoder could recover from the logits.

**Next design:** learn to nominate each local clause as one coherent packet without forcing its tokens to predict a record number. Then compare a separate grouping step that joins packets by their selected action and item, independent of mention order. The first learned head should name a whole clause before the system tries to decide whether it belongs to record one or two. Keep the immutable CLEAN-007 gate and compare on fresh unseen orders. The current evidence does not establish that record-neutral proposals will solve it.

## Trained results

| Seed | Labels | TRAIN exact route | Familiar exact cases | Novel exact cases | Novel phase pairs | Novel anchor flip | Novel anchor invariant | Novel exact quartets | Novel exact requested roles | Novel structurally valid packets | Novel abstentions |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20371 | full | 489/6,144 (7.96%) | 37/3,072 (1.20%) | 0/3,072 (0.00%) | 0/1,536 (0.00%) | 0/768 (0.00%) | 0/768 (0.00%) | 0/768 (0.00%) | 0/3,072 (0.00%) | 0/3,072 (0.00%) | 3,072/3,072 (100.00%) |
| 20371 | sparse | 263/6,144 (4.28%) | 17/3,072 (0.55%) | 0/3,072 (0.00%) | 0/1,536 (0.00%) | 0/768 (0.00%) | 0/768 (0.00%) | 0/768 (0.00%) | 0/3,072 (0.00%) | 0/3,072 (0.00%) | 3,072/3,072 (100.00%) |
| 20471 | full | 877/6,144 (14.27%) | 150/3,072 (4.88%) | 0/3,072 (0.00%) | 0/1,536 (0.00%) | 0/768 (0.00%) | 0/768 (0.00%) | 0/768 (0.00%) | 0/3,072 (0.00%) | 0/3,072 (0.00%) | 3,072/3,072 (100.00%) |
| 20471 | sparse | 896/6,144 (14.58%) | 64/3,072 (2.08%) | 0/3,072 (0.00%) | 0/1,536 (0.00%) | 0/768 (0.00%) | 0/768 (0.00%) | 0/768 (0.00%) | 0/3,072 (0.00%) | 0/3,072 (0.00%) | 3,072/3,072 (100.00%) |
| 20571 | full | 466/6,144 (7.58%) | 10/3,072 (0.33%) | 0/3,072 (0.00%) | 0/1,536 (0.00%) | 0/768 (0.00%) | 0/768 (0.00%) | 0/768 (0.00%) | 0/3,072 (0.00%) | 0/3,072 (0.00%) | 3,072/3,072 (100.00%) |
| 20571 | sparse | 631/6,144 (10.27%) | 2/3,072 (0.07%) | 0/3,072 (0.00%) | 0/1,536 (0.00%) | 0/768 (0.00%) | 0/768 (0.00%) | 0/768 (0.00%) | 0/3,072 (0.00%) | 0/3,072 (0.00%) | 3,072/3,072 (100.00%) |

## Cause-localization diagnostic

| Seed | Labels | All four exact clause packets in the right assigned slots | All four exact packets, allowing arbitrary reordering | Rows with no complete correct clause packet | Complete strict novel families |
|---|---|---:|---:|---:|---:|
| 20371 | full | 0/3,072 (0.00%) | 41/3,072 (1.33%) | 0/3,072 (0.00%) | 0/48 (0.00%) |
| 20371 | sparse | 0/3,072 (0.00%) | 7/3,072 (0.23%) | 0/3,072 (0.00%) | 0/48 (0.00%) |
| 20471 | full | 0/3,072 (0.00%) | 88/3,072 (2.86%) | 0/3,072 (0.00%) | 0/48 (0.00%) |
| 20471 | sparse | 0/3,072 (0.00%) | 12/3,072 (0.39%) | 0/3,072 (0.00%) | 0/48 (0.00%) |
| 20571 | full | 0/3,072 (0.00%) | 48/3,072 (1.56%) | 0/3,072 (0.00%) | 0/48 (0.00%) |
| 20571 | sparse | 0/3,072 (0.00%) | 78/3,072 (2.54%) | 0/3,072 (0.00%) | 0/48 (0.00%) |

The order-free diagnostic compares all 24 permutations of the four hard-predicted packets against exact role positions without using score or choosing a different production output. It distinguishes a mere permutation of complete clauses from missing or internally incorrect hard role selections; it cannot prove information was absent from all logits or activations.


## Paired annotation-density differences (sparse minus full)

| Held view | Seed | Exact-case differences, pp | 95% paired-root interval, pp | Sparse-only / full-only correct cases |
|---|---|---:|---:|---:|
| held-familiar | 20371 | -0.65 | [-2.28, +0.65] | 15 / 35 |
| held-familiar | 20471 | -2.80 | [-5.57, -0.20] | 31 / 117 |
| held-familiar | 20571 | -0.26 | [-0.75, +0.10] | 2 / 10 |
| held-novel | 20371 | +0.00 | [+0.00, +0.00] | 0 / 0 |
| held-novel | 20471 | +0.00 | [+0.00, +0.00] | 0 / 0 |
| held-novel | 20571 | +0.00 | [+0.00, +0.00] | 0 / 0 |

Intervals repeat the fixed panel by resampling whole paired roots (10,000 draws). The annotation-sufficiency claim requires both arms to pass the absolute gates across all seeds before any relative conclusion.

## What was tested

A new learned locator sees raw passage words and a question. There are no separators or token masks marking the four clauses or the two records. Twenty learned heads can each select any observed passage token, with two additional heads for the query. The finite composer sees only the selected raw word IDs and positions. It checks the two candidate action/item identities, their anchor membership and selected witness geometry, and determines the exact event and item position by an engineered rule. A repeated item name at the wrong source does not count. This is not a learned composer or a free-generation test.

Three paired seeds use the same initialization, architecture, optimizer, scheduled whole-root batches and 384 updates. Both variants see the same 6,144 unique training rows and 49,152 repeated row presentations. Full uses 1,081,344 repeated named location labels. Sparse uses exactly 270,336, or one quarter: five or six labels on every row, with fixed parity-balanced coverage over each head and every two binary factors. Unselected labels do not enter the objective, but the shared TRAIN corpus retains those labels for a no-update audit. Neither variant is answer-only, and the setups do not prove identical gradient effects or backward FLOPs.

The two held-out views share 48 disjoint roots and the same facts, words and questions. Familiar passages use block or alternating mention order; novel passages use a nested order never fitted, first record, second record, second record, first record. Whole token bags and item answers match row by row. Words and local templates were known: this is an unseen arrangement of familiar elements, not cross-grammar natural-language transfer. The rejected data draft changed irrelevant adverbs when factual or order bits changed. Those adverbs were frozen per canonical leaf before fit. On the final data a pure bag-of-words system, even awarded the best held answer for each passage bag and exact question, can solve at most 1,536/3,072 event indices and 2,112/3,072 answer nouns in either held view. Exact source and item position has a lower 604/3,072 such ceiling on the novel pane. These are information bounds for order-blind inputs, not actually fitted controls.

CLEAN-007 remains an archived reference. Because both the model and corpus change from CLEAN-007, no difference across studies estimates the effect of removing delimiters. The causal training comparison here is only the label density treatment within this exact design.

## Evidence and limitations

Training capsules carry TRAIN only. Model evaluation uses isolated filesystem/PID capsules with no held labels. They are not network-isolated. The immutable raw output ledgers and actual six trained fits and six blind scoring-source actors are admitted before separate ground-truth scoring. Same-seed contrasts are paired on roots, never on independent expanded rows. Multiple seeds share all held roots. The finite reference parser is a fixture check, not a learned competitor or model fallback. Query before/after, anchor-flip and anchor-invariant pairs require both source-and-item decisions right. The fixed-packet cue inversion and action/trigger position-swap implement the same algebraic control, not two independent learned wins.
