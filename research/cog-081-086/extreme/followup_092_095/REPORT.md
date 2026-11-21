# COG-ATTN-092–095 — Research report, November 21, 2026

**Executive result.** The conditional maze skill gate and local correction failed to improve observed competence or latency; Sudoku outcome-selection recovered a fixed MRV search rule; chess learned a tiny action-order bias but was much slower. The meaningful scoped discovery is in sliding-puzzle search: 19 self-solved training episodes supplied a compact reusable tie-breaker, reducing primitive expansions and **measured total search CPU time at identical correctness** once redundant computations were eliminated.

## Primary measurements

| Condition | Exact / untrained | Experience-based | Evidence status |
| --- | ---: | ---: | --- |
| Maze 092 (52 source+shift tasks, x1 budget) | 36/52, 20,850 moves | learned & gated 38/52, ~19,872 moves | No gate-specific gain; gate slower |
| Maze 092 (x2) | 44/52, 29,952 moves | 46/52, 25,916 moves | No gate-specific gain |
| Sliding 093b (25 held tasks) | 22/25, 74,094 expansions, 535.009 ms median | 22/25, 49,927, 422.472 ms | Matched success; better batch time after fusion |
| Sliding 093c (32 fresh exact-depth tasks) | 32/32, 38,950 expansions, 213.840 ms | 32/32, 18,699 expansions, 110.241 ms | Replicated scoped positive |
| Sudoku 094 | MRV 20/20, 1,112 nodes | learned selector 20/20, 1,112 nodes | Selector copied best fixed rule |
| Chess 095 | 20/20, 142,712 attempts, 543.99 ms | 20/20, 140,523, 1,530.65 ms | Skill-order overhead excessive |

Shuffled sliding control: 73,723 expansions / 575.108 ms in 093b, and 37,944 / 223.670 ms in 093c. Novel 8-puzzle positions were distinct from the 210-case benchmark and from each other. 22/32 fresh puzzles gained in search expansions, 10 tied and none worsened.

## Discovery and limitation

The observed mechanism is **experience-conditioned tie ordering that preserves the admissible A* scoring function**. A direct implementation made 093 slower due to extra feature processing. A fused implementation reduced that overhead. This supports an engineering principle: the useful predictor must be placed where its cost is lower than the work it prevents. A separate 238.894 ms training/setup bill was measured; approximately three 32-task batches under the same local conditions would cover it. No universal inference-scaling law is established.

The architecture has **not** demonstrated how to learn the A* heuristic, Sudoku rule invariants, full legal chess, or global open-ended planning from experience. Future work must test whether COG-ATTN can autonomously learn those operators, and avoid calling code-assisted search a neural cognition breakthrough.

**Recommendation:** Preserve 093b/c as a validated, limited action-selection optimization; refuse promotion for 092, 094 and 095. Next attempt should remove algorithmic scaffolding with a goal-and-feedback-only agent, compare against its own no-memory ablation, and target level 10 maze and deep 15-puzzle generalization with memory growth and full-cost accounting.
