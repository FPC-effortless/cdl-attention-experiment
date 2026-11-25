# COG-BIND-001W: phase supervision makes temporal binding learnable, but one seed still fails

**Two of three phase-supervised local fits passed the prospective gate. The study did not qualify across seeds.** On novel same-cue clauses, the added event-phase labels raised strict before/after pair success from **4/3,072 to 2,066/3,072** across the three matched seeds. The separate phase-supervised reference with a small per-event attention block scored **580/3,072**. These totals aggregate repeated seed observations of the same families, not independent draws. This is a bounded mechanism result, not a general attention replacement.

| Seed | Outcome-only local strict pairs (/1,024) | Phase-supervised local strict pairs (/1,024) | Phase-supervised segment attention strict pairs (/1,024) | Local phase-supervised exact rows (/2,048) | Local phase gate |
| --- | ---: | ---: | ---: | ---: | --- |
| 20371 | 0 | 972 | 566 | 1,996 | Pass |
| 20471 | 0 | 932 | 4 | 1,954 | Pass |
| 20571 | 4 | 162 | 10 | 876 | Fail |

The local phase intervention raised strict pair accuracy by 94.92, 91.02 and 15.43 percentage points (respectively, family bootstrap 95% intervals [92.58,97.07], [87.89,93.95], [11.13,19.92]). No previously successful phase pair was lost in these matched comparisons. Strict quartets requiring both before/after answers under both item assignments were **476, 444 and 52 out of 512** in the phase-supervised local models. The full 16-variation families were **110, 98 and 2 out of 128**. Familiar exact accuracy did not regress against the matched local baseline in any seed.

## Where the remaining errors occur

- **Event choice improved in all three local fits.** With phase labels, the local model chose the correct event on 2,048, 2,048 and 1,868 of 2,048 novel queries. The two stronger seeds' remaining 52 and 94 errors were all at exact item selection. In the weakest seed, 180 rows chose the wrong event and 992 had the right event but wrong item.
- **The weak model still missed most items when the correct event was supplied to the scorer:** 920/2,048. With this control the two stronger phase-supervised local fits score 1,996 and 1,954. At this point simply fixing the event route cannot recover the missing accuracy.
- The small attention reference chose the right novel event in 2,004, 2,048 and 2,016 rows. Its supplied-event item counts were 1,542, 176 and 112. This comparison does not show an inherent attention handicap; its fitted local pointer failed to transfer as reliably on this fixture.
- Outcome-only local final training minibatch event NLL was near log(2) in two seeds and about 0.003 in the third; all three phase-supervised local final minibatches were at or below 4.2e-7. The outcome-only failure is not solely a held-syntax failure. Binary phase matching has a zero-signal regime when both event phase vectors are identical or the query distribution is exactly balanced; phase label gradients are a plausible way around that regime, not proof that all failures had that cause. Fixed phase names are not identified by the outcome-only objective.

## What the comparison tests

The same person handles two different items, one before an anchor event and one after. Each raw record is queried both ways without rewriting it; strict pair success requires the right source event and exact item for both questions. The 16 rows per family cross question phase, actual timing, object assignment and physical clause order. The held sets reuse learned local frames, but change how they are combined: every novel record uses the same timing word in both clauses, once with the action as the reference and once with the anchor as the reference. That defeats literal word matching. A finite grammar control reaches 2,048/2,048 in the novel panel; this validates the constructed task, not open-language prediction. The two novel frame combinations are tied to question surface forms, so their stratum results do not isolate those two surface changes. Familiar and novel panels are aligned views of the same 128 withheld families, not independent samples.

Both local arms use the same character-composed convolutional model, 15,379 parameters, and a two-phase matching route that directly controls native output. The only local treatment difference is a 0.35-weighted training loss with before/after labels on document events; the question phase is learned indirectly in all arms. The independent segment-attention reference has 15,337 parameters and the same phase training labels as the treatment. It is a sample-budget comparison, not a full-context Transformer or equal-FLOP claim. All nine fits use 512 optimizer updates and 65,536 answer outcomes each; phase-supervised fits receive 131,072 extra document labels. At inference the actors are given only raw text and the literal comma-and boundary. Neither oracle event IDs nor candidate item masks enter model inputs. Oracle-event item extraction, given only to the scorer, isolates the cost of a wrong route. Each separated event has one `a ITEM` construction, so this is not a difficult within-event noun-disambiguation benchmark.

The original Forge diagnostics, native TrainLab budgets, inert end-to-end actor smoke, and private Research Lab provenance document execution and custody. They do not qualify it for natural language, long context, or speed claims.

## Decision and next experiment

Keep the supervision result; do not promote the complete reader. Focus the next comparison on reliable item selection under a fixed or scorer-supplied event, and retain the native phase route as a separate measured endpoint. Use counterbalanced noun roles, including a second equally plausible noun in the same event, then retest novel syntax with no oracle at inference. A stronger reader must improve the weakest seeds without trading away observed phase routing; a cross-seed pass on fresh cases is still required.

All nine native fits met their fixed-budget execution checks. The independent recount verified all 27 sealed actor bundles and matched the gates and error partitions. The private Research Lab recorded 20 evidence nodes and 28 provenance links. Recount SHA-256 `256252a6a66e541a7f36131a70fa13be7fe51a3b6a6acc31692b60cc9dcd99bf`.

Data pin `e9bee88b144dd6b207ff7a1b310355105ad075605aef97582917791b185880c8`; source lock `3fa87ee3f853510b44b32f044a4d5c5889512f7a1610de6a3b75ebb9230499f0`; score `77cdd858ed4638e02a7f8ee3232192bb8ca9ee29badab4fdc1215d1ae8174cbf`. The private replay retains all exact inputs, outcomes, raw predictions, checkpoints, and receipts.
