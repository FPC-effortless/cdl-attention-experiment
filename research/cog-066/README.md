# COG-ATTN-066: acquired procedural chunks did not reduce end-to-end cost

## Result
A learned dictionary of 16 short action sequences matched tuned exact primitive A* in outcome and primitive cost on every completed case. It did not reduce the total cost of planning and verifying work. Its typical paired workflow-family latency was **2.43×** the primitive reference, so the preregistered promotion gate failed.

This was a local CPU diagnostic on synthetic parcels, not an autonomous live task, language model benchmark, or qualification. It does not establish that hierarchical skills generally fail.

## What was checked
- **330 eligible training workflows:** all succeeded on replay and matched an independent, unpruned Dijkstra path-cost oracle. Their successful primitive traces yielded 16 chunks. No task IDs, named macro boundaries, protected TEST data, or held trajectories were used for acquisition.
- **546 held development workflows:** 182 authorized static, 182 initially blocked, 182 with a visible authorization revision. Six policies produced 3,276 outputs. The independent verifier replayed all recorded actions and receipts. All 182 authorized reference path costs matched the separate Dijkstra oracle.
- **5,166 timed calls:** 41 paired families × 3 conditions × 6 policies × 7 interleaved repeats. Each call included planning, replanning where needed, primitive application, receipt verification, and identical deterministic reporting. The timed outputs matched previously frozen outputs. All policies were uncached across calls and revisions.

## Comparison
| Policy | Verified completions / 546 | Median paired family latency ratio vs primitive | Mean extra time per workflow |
|---|---:|---:|---:|
| Primitive A* | 182 | 1.00× | reference |
| Learned chunks | 182 | 2.43× | 257 µs |
| Fixed chunks | 182 | 2.48× | 267 µs |
| Shuffled chunks, 6501 | 184 | 2.44× | 257 µs |
| Shuffled chunks, 6502 | 184 | 2.51× | 265 µs |
| Shuffled chunks, 6503 | 185 | 2.53× | 269 µs |

The paired family-bootstrap 95% interval for mean added learned latency was **212–306 µs** on this CPU workload. The learned policy attempted 499,404 primitive transitions versus 85,942 for primitive A*, about **5.81×** as many. It finished 601 chunk instances, so the result is not explained by an unused dictionary. The 12:1 inference-weighted lifecycle model was **1.674 seconds higher** at the prespecified 512 workflows. This is a weighted cost difference, not elapsed wall-clock time for the experiment.

The few extra verified completions in shuffled arms came from revision cases where a different valid plan order finished the affected parcel before the injected event. The preregistered field named `quality_regressions` counts any per-case status mismatch, including these favorable differences. These are real simulated outcomes but do not establish a robust advantage for the shuffled dictionaries.

## Interpretation
Keep the exact primitive planner as the reference for this workload. Counting fewer high-level decisions, or finding chunks used repeatedly, does not establish cheaper verified execution. An improved implementation would need a cheaper applicability filter or compiled transitions and a fresh matched comparison, including freshness checks, without excluding the actual application and verification cost.

## Handoff
- [Frozen science protocol](cog_065_066_packet/PROTOCOL.md)
- [Aggregate results](AGGREGATE_RESULTS.json)
- [Scientific source and invariant fixtures](cog_065_066_packet/)
- [Separate COG-ATTN-065 design proposal](cog_065_066_packet/COMMUNICATION_PROPOSAL.md)

The included source is the unchanged scientific implementation and protocol, with published byte hashes. From this directory, run `python3 -m unittest -v cog_065_066_packet.test_fixtures` for invariant fixtures. This public handoff does not include raw traces, private review receipts, scoped case files, internal runner configuration, or timing rows. The protocol's native execution commands describe the original full custody packet and cannot reproduce native execution from this public subset alone. Native Forge, TrainLab, Veritas, and Research Lab were used; the result remains NOT_QUALIFIED.
