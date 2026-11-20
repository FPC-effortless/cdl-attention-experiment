# COG-065: the short channel bound; a learned sender did not earn a trial

**Decision: STOP learned ranking for this setup.** At 16 words including EOS, the exact same-information codec transferred useful facts, but could not give enough facts to every recipient. The full-budget codec solved all the constructed cases. Cached exact selection already met the preregistered local CPU cost bound by a wide margin. A learned ranker cannot beat exact expected utility inside this shared finite codec and fully specified prior; none was fitted or benchmarked.

| Message policy | Correct local outcomes / 576 | Accuracy | Mean words incl. EOS |
|---|---:|---:|---:|
| No message | 216 | 37.50% | 1.00 |
| Exact same-information, maximum 16 words | 286 | 49.65% | 6.76 |
| Exact same-information, maximum 64 words | 576 | 100.00% | 15.03 |

The 16-word result improved on no message by **12.15 percentage points**, with a paired whole-group 95% interval of **10.24 to 14.06 points**. Full improved on the 16-word exact bound by **50.35 points**, interval **46.18 to 54.86**. Both preregistered lower-bound thresholds passed. In the full arm 90/144 messages exceeded 16 words, with a longest of 29. Its mean is under 16 because the exact policy sometimes needs no message; the bound restricts individual cases, not average traffic.

The 16-word codec produced 82/192 correct operator, 102/192 auditor and 102/192 requester outcomes (no-message 72/192 each; full-budget 192/192 each). There were zero unsafe simulated action outcomes under the preregistered check in each arm. This means zero **observed** cases under these rules and cases, not a population safety claim or a natural-language evaluation.

## Where the gap came from

The sender optimized jointly over four known hypothetical recipient context types, not the realized private context. The exact same four types were evaluated for every message. The reference and paired perturbation in each independently seeded group shared the same source schedule and receiver patterns. The three strategies differ only in the textual channel budget. Both nonempty strategies maximize the same exact objective and are grounded in identical causally available evidence. The full arm is not a hidden-label oracle.

The simulated timeline admitted envelopes by availability and slot position **before** semantic parsing, rejected distinct conflicting eligible same-tick positions, and never semantically parsed superseded or unavailable malformed payloads. The verifier independently interpreted full evidence for its next-action reference and replayed the unchanged primitive transitions for completion/resume. This is a bounded simulated workflow with trusted envelope order and structured receipt atoms; it does not establish authentic signing, general receiver behavior, unknown schema adaptation, or learned communication.

## One-host cost

| Local region | Samples | p50 | p95 |
|---|---:|---:|---:|
| Warm exact cache-key and lookup | 720 | .001022 ms | .001933 ms |
| Complete warm occupancy, exact selection, parsing and local verifier replay over four contexts | 720 | .054027 ms | .072350 ms |
| Cold uncached chooser only | 12 | 1.593882 ms | 2.518501 ms |

One finite 864-entry cache took **1.280 s** to compile and serialized to **116,826 bytes** (not process RSS). Warm timing excludes process load, cache compilation, serialization and network, and cold does not measure a full request. These are software paths on one host, not reproducible hardware guarantees or a ranking between learned and exact latency. The observed warm costs pass the frozen .25 ms lookup and 1 ms complete ceilings. Keep exact selection. If more supported facts are needed under this workload, budget or grounded grammar is the tested bottleneck. Revisit learned selection only with a separate variable-schema/context study where cheaper exact optimization no longer solves the practical problem.

## Evidence and limitations

- 72 constructed groups; 144 source variants; four matched recipient-context evaluations each. No external activity, paid service, protected TEST or learner.
- Independent exact preflight preceded each separate stage on the pinned local Veritas ecosystem, commit `eec439492407b0aa26922c23b5ae40d95289f4eb`. Forge froze held observations separately from the evaluator reference. TrainLab ran the deterministic emitter (1.458 s), then the Veritas-style adapter verified saved output (0.320 s). Both integrity checks passed; immutable Research Lab evidence retained.
- The artifacts are **not qualified**. These are distinct causal intervention examples within a constructed family of rule-driven local workflows, not evidence of language acquisition or open-world decisions.
- This branch contains source, protocol and aggregate results only. It excludes the underlying generated observations, private recipient context, row-level emitted messages/timings, review tokens and private evidence logs.
- Separate outcome audit independently enumerated all 864 cache keys and up to 128 legal clause masks each, checked exact optima at both budgets and all 144 emitted choices, replayed chronology and primitive traces, and reproduced both paired intervals, counts, timing summaries and STOP. No emitter, native verifier or learner was rerun.
