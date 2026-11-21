# COG-ATTN-087–090 · State-conditioned skill selection

**The selector retained optimality but failed its economic gate.** Across 144 fresh tasks it saved eight primitive evaluations and one heap pop relative to Exact A*. Including applicability, lookup, validation and final replay, it took 81.01 ms versus 62.77 ms. Reusing the primitive successors that A* had already enumerated was the fastest control at 24.71 ms. These are single-host CPU sums of per-task median timings, with three interleaved passes per task. [Frozen protocol](PROTOCOL.md) · [Independent arithmetic](results/summary.json)

| Arm | Optimal solutions | Primitive evaluations | States popped | Charged units | Online wall time |
| --- | ---: | ---: | ---: | ---: | ---: |
| A · Exact A*, 086-style | 144/144 | 17,587 | 5,630 | 19,282 | 62.77 ms |
| B · Unconditional 066 options | 144/144 | 161,972 | 5,685 | 163,667 | 343.81 ms |
| C · Length-matched shuffled options | 144/144 | 161,326 | 5,707 | 163,021 | 338.97 ms |
| D · Development-gated options | 144/144 | 17,587 | 5,630 | 19,282 | 64.61 ms |
| E · State-conditioned selector | 144/144 | 17,579 | 5,629 | 20,719 | 81.01 ms |
| Always attempt certified structural skills | 144/144 | 17,823 | 4,528 | 35,680 | 78.40 ms |
| Efficient primitive A* | 144/144 | 17,587 | 5,630 | 19,282 | 24.71 ms |

The time ratio A/E was 0.775×. A 5,000-draw paired bootstrap over the six seed blocks gave a 95% interval of 0.746–0.793×. This small-seed interval describes the measured host and fixtures. Charged units add one unit per primitive attempt, synchronous topology read, revision comparison, selector query, and final replay step. They are a proxy. Successor enumeration, dictionary work and feature extraction also appear in the wall clock. Raw outputs separately record generated neighbors and neighbor calls. One-time training and development took 257 ms, excluded from online totals. [Held tasks](results/held.json) · [Replication tasks](results/rep.json)

## 087 · What was learned

The runner collected 237 demonstration records from 72 training tasks: 49 corridor bottlenecks and 188 goal-finishing suffix examples. It stored 102 distinct sliding-puzzle and 60 Hanoi suffix keys. The corridor and suffix schemas, finite primitives and proof obligations are provided. The subgoal locations, normalized sequences, applicability scores and expected remaining search savings come from training. Maze edges contract only degree-two interiors, stopping at junctions, dead ends or goals. A learned board suffix may replace outgoing branches only after a legal primitive replay reaches the current goal at a certified lower bound. For three-peg tower transfers the lower bound can use the minimum number of moves for the smallest sub-tower. Four-peg cases use a weaker safe bound.

Two depth-four trees were fit to 497 paired training rollout states. Each target compared the remaining Exact A* primitive evaluations with a counterfactual invocation at that state alone. Suffix demonstrations from the same seed were excluded from its training target. Applicability estimates whether the invocation produces a certified reduction of the successor set. The online gate requires predicted applicability of at least one half and predicted savings above the development threshold. Legal neighbors and available goal data remain public to every arm. A missing or invalid suffix falls back to exact search. [Training pairs and effects](results/training.json) · [Locked model](results/model.json) · [Pre-test lock commit](https://github.com/FPC-effortless/cdl-attention-experiment/commit/efcf17f)

## 088 · Why the selector missed the speed gate

All three domain-level 066 option sets were rejected on 48 development tasks. B and C made about 9.2 times as many primitive evaluations as A on this new distribution. The prior 4.29× figure describes a different, seven-condition panel. In E, development disabled maze contraction and selected a 0.5-evaluation savings threshold for sliding and Hanoi. E invoked 19 times and obtained five effective certificates. The disable threshold still incurs lookup and scoring overhead in this implementation.

A post-test diagnostic sampled 416 starting and intermediate states, with 144 eligible proposals. Only 38 forced first invocations saved any primitive evaluations. Just three saved charged units. Among 137 maze proposals, 34 reduced primitive evaluations but none repaid the topology and revision charges. Only one sampled sliding state and six sampled Hanoi states had a matching demonstrated suffix. E selected seven of the 416 states, four of which saved evaluations. These are local counterfactuals; their savings cannot be added to predict whole-search speed. The two separate failure modes are the cost of certifying corridors and the scarcity of applicable suffixes on unfamiliar goals. [Paired-state diagnostics](results/paired-state-diagnostic.json)

## 089 · Shifts, partial maps and validity

The runner regenerated the same 36 maze shifts as PR #75 and checked the before/after lengths against its archived results. All 36 original plans became invalid. All were refused by the revision-bound reader, and all 36 replans matched independent shortest distances. Across those changes the dependency guard retained **4,446** unaffected directed corridor records and rejected **524** affected ones. It also caught 36 off-path wall openings that changed a corridor precondition even though the original moves could remain legal. Changing a goal invalidated the old goal-bound plan while leaving unrelated local geometry usable. A goal inside a cached corridor terminated the skill early.

With a dependency hidden, the reader refused the cached skill in all 36 cases. In 18 additional masked-dispatch runs, 3 tasks reached the goal using only confirmed-open cells. The other 15 deferred movement and reobserved a total of 135 masked cells before planning. All 18 reached the new goal with zero stale dispatches. These checks exercise local single-process version visibility. The maze skill was disabled by E's development gate, so its integrity results are separate from E's timing. [Version checks](results/shifts.json) · [Masked execution](results/partial.json)

## 090 · Evaluation and replication

The held and replication panels each have 72 tasks from three seeds. Each seed supplies eight tasks per domain. All 144 complete fixture digests are distinct and separate from the 120 training/development fixtures. Maze topology and goal vary together. There are 40 distinct maze goal coordinate pairs, 47 distinct sliding goal vectors and 46 distinct Hanoi goal vectors across the two panels. Held sliding goals use a reserved blank-position stratum. Held board goal vectors are absent from training and development. Half the 48 held and replication Hanoi tasks use four pegs, unseen during training. Six seed blocks do not establish broad distributional generalization.

A separate transition enumerator and breadth-first search checked submitted actions and optimal distances. The learner did not receive held reference paths or distances. The recorded runs used Veritas Research Lab OS and TrainLab. A separate GitHub-hosted CPU reproduced frozen-model decisions and work. This is machine replay with shared source; it does not establish independent-investigator replication or integrated chess/Sudoku competence. [Lab receipts](results/lab.json) · [Hosted replay](https://github.com/FPC-effortless/cdl-attention-experiment/actions/runs/44843337374)

## Exploratory subgoal repair

A separate, frozen Hanoi experiment allowed an abstract subgoal to alter frontier ordering. It computed an admissible distance for the smallest three discs with the larger discs removed, and charged cold table construction per puzzle. A new tree was trained on paired remaining workloads and development could disable the mechanism entirely. On 60 fresh tasks, always consulting the bound reduced main-search transitions from 49,620 to 41,267. Building the tables required 8,928 transitions, bringing the total to **50,195**, above the cheap misplaced-disc heuristic's 49,620. Times were 75.54 and 75.39 ms respectively. Development disabled the gated mechanism, which reproduced the cheap heuristic's work. No learned speedup qualified. This secondary comparison preserves the primary result and identifies the concrete amortization hurdle for a persistent subgoal index. [Supplement protocol](SUBGOAL.md) · [Subgoal model](results/subgoal-model.json) · [Fresh results](results/subgoal.json)

## Reproduction

Use Python 3.12 from this directory. Frozen inference and the independent audit require only the standard library and the sibling 081–086 sources. To refit the trees, install NumPy 2.3.5 and scikit-learn 1.8.0. Model selection in the supplementary fit includes development timings and may vary by host. Use a fresh checkout when regenerating measurements.

```bash
sha256sum --quiet -c SHA256SUMS
python replay.py
```

For a full rebuild, run train.py first, then panel.py held and panel.py rep. Run shifts.py, partial.py, diagnose.py, subgoal.py fit, subgoal.py score and audit.py. To reproduce the Veritas path, set SKILL_LAB to a fresh directory, put the research_lab and trainlab src directories from Veritas revision eec439492407b0aa26922c23b5ae40d95289f4eb on PYTHONPATH, and run lab.py. The two jobs verify their TrainLab artifacts and register their evidence pins with Research Lab OS. Original model fitting and post-test diagnostic scripts are separate from the held inference entry point. [Veritas source](https://github.com/FPC-effortless/veritas/tree/eec439492407b0aa26922c23b5ae40d95289f4eb)
