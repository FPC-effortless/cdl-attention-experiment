# COG-BIND-001S: role supervision does not resolve new-syntax event-object binding

## Result
Adding explicit token-role supervision to the same reader did not fix binding. Both arms answered familiar wording at approximately 99–100%, but the added-supervision arm got only 239, 258 and 256 of 512 novel both-exact item-swap pairs across the three seeds. No seed passed the prospective gate. Neither arm got all eight variations of any of the 128 novel families correct. A deterministic proximity comparator outperformed both learned arms on novel swap pairs.

| Seed | Familiar exact, answer-only → role supervision (of 1,024) | Novel item-swap both exact (of 512) | Novel same-clause changed-question both exact (of 512) | Novel event-order both exact (of 512) |
| --- | ---: | ---: | ---: | ---: |
| 16371 | 1,015 → 1,014 | 246 → 239 | 0 → 4 | 241 → 235 |
| 16471 | 1,022 → 1,024 | 247 → 258 | 0 → 8 | 238 → 256 |
| 16571 | 1,023 → 1,023 | 256 → 256 | 0 → 5 | 256 → 256 |

The role-supervised renderer cells were 117/256 and 122/256; 128/256 and 130/256; 128/256 and 128/256. The independently bootstrapped, within-seed family-clustered 95% intervals on paired accuracy gain were [−3.32,+0.59], [+0.98,+3.32], and [0,0] percentage points. One seed improved and the result was not repeatable across seeds.

A post-hoc recount isolates the predominant failure further. On the 512 novel rows per model that ask about the event worded in passive voice with an intervening modifier, answer-only got 0/512 in every seed; the role-supervised arm got 4, 8 and 5. When the question selects the active event, the corresponding counts are 497, 494 and 512 (answer-only) and 490, 512 and 512 (role-supervised). The models often point outside both item spans on the failed passive cases; they do not simply copy the active item regardless of question. Voice, intervening modifier and other novel wording changes were not separately randomized in this experiment. These results localize a failure to retrieving the payload within the supplied utterance, but do not identify which linguistic change causes it.

## Design and scope
001R localized failures to interpreting the item even when the supporting record was correct. This study therefore gives the reader one correct supporting utterance and a raw question, with two legitimate events in the utterance. Half the families require distinguishing verbs with the same actor; the others require distinguishing actors doing the same action. A matched 12,927-parameter query-conditioned character-composed convolutional pointer used either answer supervision alone or the same loss with a token-level role objective (queried item, other event's item, other tokens). The architecture, initial weights within each seed, optimizer, training rows and order were matched. The added supervision was applied only during training. The reader uses bounded soft communication between clause and query, so this is not an attention-free model comparison.

There are 256 training, 64 development and 128 held semantic families, eight rows each. The familiar and novel held views share those same 128 families and 32 held item spellings. Item spellings did not occur in training rows; raw spellings are composed from characters and the copied token is whatever appears at the predicted position. The spelling inventory was shared, with no per-word learned weight or item-only candidate mask. Each family crosses same-word item swaps, same-clause query flips, and event-order reversals. The novel view combines mixed voice, different conjunction placement, an intervening passive modifier, and a query timing marker. Novel renderer denominators are 256 pairs apiece; seeds share the same families and are not extra independent test cases.

The first corpus draft was stopped before fitting because a nearest-event-cue heuristic got every novel answer right. On the frozen revised corpus, the stronger nearest-unique-cue comparator gets 768/1,024 rows and 384/512 swap pairs, but 0/128 full families. First/last item each get 512/1,024 rows. The explicitly engineered typed grammar reference is exact on the entire constructed corpus. This makes the test useful for rejecting shortcuts, not a claim that the language cannot be solved with rules.

The prospective role-supervision gate is 90% novel paired exact macro accuracy, 80% minimum in each novel renderer, and no more than two points of familiar individual degradation against answer-only, in each of three seeds. A training fit sees only TRAIN labels; prediction and official scoring are separate. This does not validate a retriever, open-language competence, real-world intervention safety or serving cost.

## Decision and reproducibility
Do not deploy this reader or add retrieval/memory capacity as a remedy for this failure. A follow-up should separate voice, modifier location and question wording, and compare learning an event match followed by span copying against this single-step pointer at matched budgets. Both held views here are exposed and cannot qualify that follow-up.

Six fits on CPU, 384 updates and 49,152 row exposures each. No reruns or alternate checkpoints. An independent parser and recount verified the raw prediction wires for all 18 seed/arm/panel combinations. This repository entry contains the aggregate only.

Data pin `90fb86cf7ec3185943632dce7230ddf7e80d016038cd44b9b5650b232857195f`; prefit lock `d6a8f4a9cb028657570073e3ae948cb475b089fc6825c2d8db9f7829919bed79`; score `a8193885a84fb9883b7130ba4a26de5efebb9631896c56f0a565a61686168941`; independent recount `1b9c67c0f4ba6a13a332d767a64bafbe834ab71e45018949457db0d733a2df05`.
