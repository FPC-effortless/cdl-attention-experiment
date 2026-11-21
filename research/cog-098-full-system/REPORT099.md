# COG-ATTN-099 · integrated controller, failed cross-domain competence

A single outcome-trained policy can now act across three environments, carry working state, stage actions, and update itself at episode boundaries. Across three unassisted fits it learned enough to navigate some empty rooms. It never solved a locked-door task, missed every change in the recorded hotel preferences, and solved at most one of fifteen held-out algorithm defects. On the three historical PySnooper defects, none of the 180 proposed edits available to the policies passed its regression. This cycle does not qualify a general COG agent or establish an advantage over self-attention.

## Capabilities exercised

The same 097e RawPolicy (531,073 trainable parameters, initialized afresh per fit) ranks raw-observation/action pairs in all three domains. The 74 Workspace holds the observation and proposed action. Actions pass revision and legal-menu checks before the environment executes them. State consists of three recent observations, within-episode state/action visit counts and an outcome graph learned on training transitions. The planner looks one transition beyond immediate reward. Each adapter supplies its action vocabulary, including a fixed hotel slot schema and a program-edit grammar. The agent cannot generate arbitrary responses or patches. One gradient update follows each episode, using the outcomes of chosen actions and discounted terminal credit in navigation. The weights and graph freeze together. The existing PR 74 layout planner, path checker, authority and versioned transaction ledger are exercised separately through the native local issue adapter. The three benchmark adapters verify outcomes through their own environments.

Each fit received 281 episodes (16 software, 217 dialogue, 48 navigation) and 281 optimizer updates. The training budget was held fixed. A new scale or learning-curve experiment is needed for an architecture-capacity claim. Separate authenticated issue-transaction tests check commit, denial and stale revision. Their success says nothing about whether the learned policy understood a ticket.

## Frozen unassisted scores

Seeds are 1701, 1707, 1711. Each column reuses the exact same cases. The attention control has 535,297 parameters and the same budget, memory and execution shell. It pools the identical hashed subword stream into blocks before self-attention. This is a small from-scratch control.

| Outcome | COG | COG with history, visit counts and graph disabled at test | Attention control | Hand-written policy |
|---|---:|---:|---:|---:|
| QuixBugs repair passes public and hidden cases, 15 bugs | 1 / 0 / 0 | 1 / 0 / 0 | 0 / 0 / 0 | 2 |
| Every user turn has both hotel preferences correct, 200 dialogues | 50 / 50 / 50 | 50 / 50 / 50 | 50 / 50 / 50 | 89 |
| Correct joint preference on 228 changed user turns | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 133 |
| Empty-Random goals reached, 24 episodes | 24 / 8 / 17 | 4 / 4 / 0 | 6 / 1 / 3 | 24 |
| DoorKey goals reached, 24 episodes | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 24 |
| Unique test layouts absent from training, both families, 39 layouts | 16 / 4 / 13 | 3 / 3 / 0 | 3 / 1 / 2 | 39 |
| PySnooper regressions repaired, 3 compatibility-validated cases | 0 / 0 / 0 | not run | 0 / 0 / 0 | 0 |

COG's one accepted unassisted repair was find_first_in_sorted. Candidates were generated from visible source with a generic single-site edit grammar. Each policy got at most five attempts on a bug and only public test feedback. Hidden QuixBugs cases were held back until scoring. Original failures, timeouts and candidate status are in the raw traces. Passing the supplied cases is bounded evidence for a patch.

In 1,669 MultiWOZ user turns, two COG seeds always predicted both preferences unspecified. Seed 1701 gave that answer on 1,668 turns. The data came from actual multi-turn conversations. The task was offline joint hotel area/price tracking. The speakers never reacted to the agent. There were 44 distinct navigation layouts among 48 test episodes, of which 39 were absent from training. COG beat the small attention control on the new-layout subset at all three seeds. The ablation jointly removes three memory features, so their individual contributions are unresolved. No unassisted COG or attention fit experienced a successful DoorKey trajectory during training. The 16 training repair tasks yielded respectively 2, 0 and 0 positive actions for COG. These reward counts limit what a negative comparison can establish about the architecture.

The persistent graph recognizes states by an exact hash of the current observation. It recognized 10 of 1,669 held-out dialogue observations per seed and zero held-out program observations. In navigation it recognized 840/1,387, 1,462/1,948 and 1,240/1,894 encountered views. Repeated egocentric views can occur in different rooms. The agent has a learned text encoder as well, so this observation localizes a limitation of the graph lookup specifically. Counts are in GRAPH_COVERAGE099.json.

## Action-space ceiling on real software

All three BugsInPy PySnooper cases were retained. Under the ordinary UTF-8 checkout only case 3 reproduced as fail-before/pass-after. Case 1 already passed on the buggy version. Case 2 could not collect its test because a test-only fixture was missing. In a separate compatibility panel, using a C/ASCII locale for case 1 and the missing test fixture for case 2, all three buggy revisions failed and all three fixed revisions passed their targeted regressions. The run used Python 3.8.20. The benchmark specifies 3.8.1. Checkpoints never received the reference implementation or patch.

After the frozen runs, the evaluator tried every offered edit. On cases 1, 2 and 3, respectively, 0/64, 0/64 and 0/52 passed the target test. A second exhaustive check ran the other 100, 100 and 0 edits removed by the shortlist. None passed. Across all three cases the full generator produced 0 test-passing edits out of 380. Neither the shortlist nor the five-attempt budget can explain this ceiling. Even an ideal selector from these menus would score zero. That result localizes a concrete intervention: expand or learn the edits an agent can propose, then measure test-passing candidate coverage independently of selection accuracy.

## Assisted run and cost

The first panel supplied an engineered domain score to both architectures. For MiniGrid it included a visible-view planner's preferred primitive. Its COG navigation totals were 26, 8 and 48. The unassisted panel removed that score and bypassed the heuristic computation during both training and evaluation. Its scores above are the primary result. Both panels and unsuccessful attempts are retained. The heuristic saw only the public view and attained 48/48 navigation episodes.

The unassisted COG fits took 32.5–34.6 seconds, and each complete 263-task test took 42.4–48.1 seconds on the measured CPU host. Attention fits took 46.8–48.5 seconds and test took 60.7–68.5 seconds. Timings include environment work, verification and scoring inside the driver. Interpreter startup and imports precede its timer. Multiple runs shared the host and action counts differ. These figures cannot establish an inference speedup at matched quality. RUNS099.json includes work counts, within-process high-water RSS, model and trace digests, and individual timings.

## Admission and evidence

All 24 local tests passed. The native transaction regression also verified denied writes, stale-version rejection, rollback and duplicate commits. Six separate frozen COG replays (three seeds in each advice condition) exactly matched the original per-action and outcome digests, and the rescorer reproduced all 19 panel runs. The original 098 gate recognizes the implementation, entry points and pinned public data. It still returns NOT_QUALIFIED because independently held evaluation receipts are absent. The panel runs, rescoring and frozen replays all took place on the same host. The dialogue task is narrower than goal fulfillment, navigation covers two variants of one environment, and the real-code extension covers one project. No evidence here supports general conversational ability, unrestricted program synthesis, effective online transfer at larger scale, or replacement of self-attention.

See CASE_SCORES099.csv for the paired per-case outcomes, RUNS099.json for model and cost records, EXTERNAL_SCORES099.json for the external candidate audit, AUDIT099.json for rescoring and replay, and REPRODUCE099.md for exact commands. The companion evidence ZIP contains all frozen weights and action receipts.
