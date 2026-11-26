# COG-UNIFIED-017: governed state and matched interpreters

2026-11-26. Developmental CPU-only study. Three seeds per arm (5209,5231,5279). Four jointly supervised interpreter/history models and two endpoint-only controls, each at 300 updates and 12 histories of 16 events per update. COG uses the recovered UNIFIED-016 block with the division-free scan. Gated delta-rule is a small reference, not official Gated DeltaNet. Joint event and final-answer labels are provided to all primary models. Tests use synthetic word IDs, explicit span markers, 8 entities and 8 values; not natural language understanding. Unseen wording = withheld span orders; unseen aliases = withheld root/modifier combinations of known words. No protected test data, paid compute, GPU, full-scale language training or canonical promotion.

## Primary held-order + held-alias results

|Architecture|Parameters|Correct event interpretations (1024 histories panel)|Neural-only complete 16-event histories|Exact store complete 16-event histories|Neural-only complete 1024-event histories|Exact store complete 1024-event histories|
|---|---:|---:|---:|---:|---:|---:|
|COG|23997|49075/49152|1/192|191/192|0/48|42/48|
|Transformer SDPA|23937|31548/49152|1/192|4/192|0/48|0/48|
|GRU|24001|49152/49152|0/192|192/192|0/48|48/48|
|Gated delta-rule|23945|49152/49152|95/192|192/192|0/48|48/48|

Familiar-wording event interpretation on 1024-event panels: all four primary arms 49152/49152. Every primary exact store solves 48/48 familiar 1024-event histories. COG's three held combined 1024-event counts: 16,16,10 of 16; it fails the every-seed >=95% gate. GRU and delta-rule pass the synthetic gate. Oracle store solves every panel. Endpoint-only controls do not train event heads, so their proposals are not treated as equivalent interpreters. The interpretation accuracy includes correct no-op action on HOLD/DEFER; a correct destination and payload are required for relevant commits (payload only for WRITE).

Confidence thresholds selected on familiar development events are 0 for every primary fit: no extra filtering. On the held combined 1024-event panel COG accepts 31520 proposals, 75 incorrect; Transformer 32533, 17541 incorrect; GRU/delta-rule 31522, zero incorrect. Familiar calibration does not control error on distribution shift. No claims of learned arbitrary ambiguity detection or authority validation.

## Replay and intervention

18/18 checkpoint metric and calibration replays exact. Across 18 checkpoints and all evaluation panels: 13,572,540 nonselected-coordinate checks, 0 violations; 1,173,060 selected writes/deletes; 0 internal/external disagreements and 0 state-restoration disagreements. Oracle: 10368/10368 evaluated history instances, including repeated histories across regimes/arms (not independent sample count). Separate checked API: 2400 exhaustive discrete test cases, 0 violations; checks types, ranges, threshold, NaN, no-op behavior and replay idempotence. These are structural, not semantic guarantees.

Fresh paired 128-event held-combination histories: COG protected store 180/192 complete pairs for first-write value change, ambiguity change, and distractor-only change; GRU and delta-rule each 192/192; Transformer 0/192 each. COG retains unrelated answers in 192/192 pairs, but correctness still fails on some histories. Neural-only COG fails all complete paired histories.

## Warm CPU timing

Same checkpoint's interpreter, same proposals, same correctness; timed two chunks, encode/decode, update, serialize and restore; counterbalanced ordering by repetition/seed, median of 3. At 1024 events, batch=1, sum of seed medians divided by correct sessions: COG tensor 63.808 ms vs dictionary 36.331 ms (3/3 correct); GRU 36.268 vs 7.446 ms (3/3); delta 50.369 vs 21.148 ms (3/3). At batch=8: COG 53.531 vs 47.507 ms per correct session (24/24); GRU 11.673 vs 8.734; delta 31.051 vs 27.455. Transformer no correct sessions on these held timing histories. Tensor logical store is 8 int64 values (64 bytes per session), dictionary variable JSON storage. This implementation is not optimized, comparisons are at same checkpoint quality; no lifecycle cost, GPU performance or process-crash claim. The internal tensor is a separately implemented governed state path, not a trained full language-model protected layer.

## Accounting and conclusion

18 distinct training fits; 5400 optimizer updates; 64,800 training histories; 1,036,800 training event presentations. Checkpoint evaluation replays and deterministic tests separate. No official campaign credit. Preserve negative results. COG's improved supervised interpretation does not confer a special advantage over simpler interpreters with the same store. Prioritize native contextual computation and independent semantic transfer. No canonical promotion.
