# COG-PULSE-004A-R2: matched full-vocabulary answer heads

**The tied value path helped individual answers, but it did not make the frozen reader reliable.** Across all three seeds, the candidate improved exact answers on cases containing a current fact, an obsolete fact and an unrelated fact. It rescued **zero additional complete roots**. Neither head met the preregistered gate, and there is no qualified successor.

## What was held fixed

The same three trained R2 content-aware readers supplied each comparison with the exact same query state and native mixed read. Their encoders, router and null mechanism did not change. Both freshly initialized heads began with identical fitted-parameter and output values within each seed; each trained 15,504 active parameters for 256 updates of 96 examples, using only ordinary full-48-way answer cross-entropy on the original 2,736 TRAIN rows. No gold record, symbolic value, or external probe entered primary inference. CAL contained 1,140 constructed cases from all 15 remaining disjoint two-owner roots in the fixed alphabet. TRAIN and CAL were physically separated for fitting. CAL is an open controlled test with supported symbols, not a protected or open-language benchmark.

The baseline was the native 112→96→48 SiLU shape, newly initialized. The candidate used the same nonlinear path and a tied linear residual on the channel-summed read. The residual used the second-layer matrix times the mean of the first layer's four read blocks, minus its frozen initial projection. Thus it contributed exactly zero at the start. It added 768 frozen floats and extra arithmetic, but no fitted parameters. The old trained head was a descriptive reference, not a matched control.

## Exact results

Numbers are in seed order **173 / 271 / 907**.

| Diagnostic | Fresh native-shaped head | Fresh tied-value head |
|---|---:|---:|
| Exact mixed-bank answers / 480 | 174 / 186 / 114 | **200 / 210 / 126** |
| Complete mixed-bank roots / 15 | 0 / 2 / 0 | 0 / 2 / 0 |
| Both answers exact when the current value changed / 240 | 38 / 62 / 18 | **48 / 68 / 24** |
| Both answers exact when only the visible timestamps swapped / 240 | 26 / 42 / 6 | **46 / 54 / 12** |
| Exact on all current-fact rows with unseen TRAIN owner–value combinations / 772 | 280 / 290 / 128 | **338 / 322 / 166** |
| Correct NONE over a fully occupied bank with no matching identity / 120 | 58 / 52 / 52 | 56 / 58 / 48 |
| Exact mixed-bank case answers with the correct record supplied at unit strength / 480 | 368 / 376 / 288 | 416 / 376 / 304 |

At normal read strength the candidate rescued 32, 24 and 16 mixed-bank answers and harmed 6, 0 and 4, respectively. It did not match the old frozen head on all seeds: the old head's corresponding totals were 188, 186 and 130. Both new heads fit all 1,152 mixed-bank TRAIN answers at their fixed endpoint, so the weakness is not a failure to fit that panel. These case counts share just 15 roots; the rows and seeds should not be treated as independent repetitions of the task.

Two controls narrow the diagnosis:

- **Supplying the correct packet helped both heads, but left many errors.** The tied head rose to 7, 6 and 3 complete roots out of 15. This bypasses selection and restores read amplitude; it is not achieved by the model. It shows that native mixtures are costly and that supplying the packet alone does not finish the unrestricted answer problem.
- **The fitted residual traded rejection for factual answers.** Turning off only that residual reduced mixed-bank exact answers from 200/210/126 to 36/68/48, but increased correct occupied-bank NONE responses to 110/106/96. Zeroing the entire read produced NONE on all 120 no-match cases in the tied arm and zero correct factual answers. Those controls do not establish learned abstention.

**Decision:** Do not advance this head as a fix. The earlier 16-value probe showed that the packet often contains the value; the full-vocabulary test shows that accessing that value and deciding whether the bank contains applicable evidence remain different problems. Next, use the retained logits and frozen routing trace to separate low target mass from wrong full-strength answers on the same roots. A subsequent matched run should test whether a query-conditioned read can preserve a selected packet's strength without turning an occupied no-match into a factual answer. Keep proper abstention and any natural-language transfer as separate gates.

All 12 native roles completed once on one CPU, within the registered 192-second ceiling. No original model parameters changed, no paid or protected run occurred, and no serving-speed claim is made.
