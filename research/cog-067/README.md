# COG-ATTN-067: information acquisition found no useful intervention signal

All seven policies chose **zero active interventions**. The calibration panel failed the preregistered joint correctness and coverage target, so the fixed fallback was used only to preserve the negative diagnostic. On 160 fresh simulated work sessions, the adaptive policy did not reduce regret and was about twice as slow as the matched fixed schedule. Both promotion gates failed.

## Controlled test
Two simulated FIFO queues shared early observations and paired future arrivals when the independent scorer compared no-op with two mitigations. The source procedures were a direct worker queue snapshot and a separate event-journal replay. They measured different early quantities but shared ingress and were **not established as statistically independent**. Neither measured the direction or strength of the intervention mechanism. A passive-exporter mirror checked redundant lineage.

The empirical table and confidence threshold were fitted separately from evaluation: 240 TRAIN sessions, 80 CAL sessions, then 160 fresh DEV sessions with 12 requested decisions each. Five interleaved timings per policy-session regenerated the same deterministic outputs. This was one local CPU implementation, with simulated deadline ticks and no external information purchases.

## Results
| Policy | Served / 1,920 | Displaced | Explicit no-op | UNKNOWN | Median session |
|---|---:|---:|---:|---:|---:|
| Adaptive value of information | 1,280 | 640 | 104 | 1,176 | 95.0 µs |
| Fixed value-of-information schedule | 1,280 | 640 | 89 | 1,191 | 47.6 µs |
| Calibrated uncertainty-first | 1,280 | 640 | 109 | 1,171 | 56.9 µs |
| Passive | 1,920 | 0 | 0 | 1,920 | 33.7 µs |
| Always ask | 960 | 960 | 284 | 676 | 142.5 µs |
| Matched random schedule | 1,280 | 640 | 109 | 1,171 | 47.8 µs |
| Query but ignore content | 1,280 | 640 | 0 | 1,280 | 47.1 µs |

- In calibration, the fixed 0.55 fallback accepted 301/3,840 proposals (7.84%) and 227/301 were correct (75.42%). No listed threshold met both gates. On development, all 1,176 adaptive proposals for active intervention became UNKNOWN.
- All seven policies incurred **8,828 counterfactual regret units**. This is a collapse to the same effective action, not evidence of comparable useful intervention competence. Adaptive decision coverage was **8.13% of served cases**, all explicit no-ops. Active precision was undefined, not measured as either 0% or 100%.
- In the five quota-matched arms, every session spent four of 12 simulated ticks on two direct observations and one replay, handled eight requests, and displaced four. The 640 displaced requests accounted for 2,620 potential improvement units in each matched arm. Passive displaced none. No pricing interpretation is implied.
- The adaptive policy added **48.6 µs per session** versus the fixed schedule (paired-family bootstrap 95% interval 47.6–49.7 µs). Real elapsed CPU-workload latency and simulated opportunity cost are separate measures.
- The oracle found **756 cases where an active action strictly improved on no-op**: 390 in the strong coupling stratum and 366 in the weak stratum. Null coupling has zero physical change and incurs the synthetic active-action charge; reverse coupling is adverse. A raw activation count would miss this distinction.

## Interpretation
This experiment does not show that adaptive evidence acquisition generally fails. For these policies, measuring congestion more carefully did not yield reliable evidence about the **effect of changing the control**. The evaluator assigned coupling by group order. Held forecast inputs exclude the coupling field, and the adaptive policy does not use group order for its decisions; this is scoped code and input separation, not a claim of formal access isolation or impossibility of inference with arbitrary features.

A better next experiment would compare a bounded, measured intervention-direction probe with a sham/no-op probe and the existing passive sources on a fresh held cohort. Establish that the candidate observed evidence can identify the sign and useful magnitude of the intervention at all, before testing whether adaptive acquisition is worth its cost. Preserve opportunity, calibration, and strong fixed controls.

## Handoff
- [Frozen protocol](cog_067_packet/PROTOCOL.md)
- [Aggregate results, effect/fault slices and risk–coverage diagnostics](AGGREGATE_RESULTS.json)
- [Code and handwritten fixtures](cog_067_packet/)

The public subset preserves original scientific source bytes and aggregate results. It excludes private scoped arrays, fitted model tables, raw traces, timing rows, review approvals, and native custody configuration. Use `python3 -m unittest -v cog_067_packet.test_fixtures` from this directory for the handwritten checks. Full native replay requires the separately held custody packet described in the protocol. Forge, TrainLab, Veritas and Research Lab were used; the result is NOT_QUALIFIED.
