# COG-UNIFIED-018 / ATTENTION-EQUIVALENCE-002 — developmental retrieval repair

Date: 2026-11-26. Source lineage: UNIFIED-017 reference/lm.py and the previous attention primitive audit. Same prepacked learned key+value+type input embeddings, 16 known identities, 8 independent randomized values, 8 distinct records and 4 queries. Answer-only cross entropy. 96 distinct fits (39 ~24K parameters, 57 ~12K), 500 AdamW updates each, LR .003, batch 24, clip 1. Primary seeds 7103/7121/7151; secondary 7193/7211/7243. FFN widths adjusted to match parameters, hidden width 32. No extra address labels or external store. Explicit query-HOLD for most associative candidates; auto arms use ordinary learned write gates without the explicit mask. All controls receive the same record/query type input. CPU, developmental only.

## Primary matched ~24K experiment
| Arm | FP32 carried bytes at 8-record prefix | Individual % | Complete four-query % |
|---|---:|---:|---:|
| Native32 |256|55.404|7.292|
| Hold32 |256|55.143|4.883|
| Slot32 |256|56.364|6.771|
| Add32 |256|91.276|70.052|
| Delta32 |256|90.723|68.034|
| Native128 |640|95.166|82.031|
| Hold128 |640|96.159|85.417|
| Slot128 |640|96.029|85.156|
| Add128 |640|99.951|99.805|
| Delta128 |640|99.870|99.479|
| Narrow SDPA width 4 |256|100|100|
| Narrow SDPA width 10 |640|100|100|
| Full SDPA four heads |2048|100|100|

All attention controls pass competence gate. Add128 and delta128 pass the precommitted basic and near-attention thresholds in every seed. Native128 has 24,365 params (1.52% above target), FF width 4; candidates around 24,015 params. The allocation of FFN parameters differs substantially; this is a state-matched architectural capacity allocation comparison.

## Secondary ~12K experiment
| Arm | Params | State bytes | Individual % | Complete % | Relevant pair % | Nuisance pair % | 16-record complete % |
|---|---:|---:|---:|---:|---:|---:|---:|
| Native32 |12009|256|55.322|7.682|3.255|5.143|2.474|
| Hold32 |12009|256|57.275|7.357|3.125|4.427|2.214|
| Add32, 8x4 |12021|256|91.781|71.615|59.766|61.133|25.911|
| Add32, 16x2 |12010|256|96.810|88.737|83.659|84.831|64.648|
| Delta32, 16x2 |12010|256|99.512|98.047|97.396|97.135|85.221|
| Delta32, 16x2 auto |12010|256|96.354|86.784|80.404|82.617|57.617|
| Add32, 16x2 auto |12010|256|98.470|94.141|91.536|91.276|65.365|
| Delta32 plain |11987|256|99.316|97.591|95.703|96.484|79.167|
| Add128 auto |11990|640|99.967|99.870|99.805|99.870|95.964|
| Delta128 auto |11990|640|99.821|99.349|98.958|99.089|89.258|
| Add128 uniform read |11990|640|44.564|2.799|0.456|1.302|0.846|
| Delta128 uniform read |11990|640|43.099|2.214|0.716|1.107|0.651|
| Add128 direct readout |11987|640|99.919|99.674|99.284|99.544|92.839|
| Delta128 direct readout |11987|640|99.886|99.544|99.219|99.284|91.016|
| Narrow SDPA |12016|256|100|100|100|100|100|
| Full SDPA |11979|2048|100|100|100|100|100|

Delta32 16x2: 6114/6144 individual answers, 1506/1536 complete histories, 1496/1536 relevant mutation pairs, 1492/1536 nuisance pairs. Per-seed complete counts 506,489,511 of 512. Passes screening thresholds in all 3 seeds with explicit HOLD. Compact auto variants not all-seed near-attention-qualified; 640-byte auto arms pass near-attention screening. Narrow attention matches compact recurrent storage at eight records and grows thereafter. Direct readout controls show no established COG-specific advantage from nonlinear operator bank. Delta and outer-product fast weights are established mechanisms; no novelty claim. Secondary experiments selected after partial primary results, new seeds and panels; not canonical promotion.

## Mechanism diagnostics
Fresh panels: delta32 16x2 98.112% complete, uniform read .130%, zero memory .065%, cyclic wrong read .065%. Uniform-read trained 128D controls also fail. Mean cross-key cosine RMS for add32 8x4 .26438 vs add32 16x2 .09213; delta32 16x2 .07321. Learned query gate means auto add128 .10379 vs record .87482; delta128 auto .07100 vs record .59237. Not exact HOLD. Repeated-key revision OOD not admitted as a comparative gate: baseline positionless attention trained on unique keys fails; delta128 auto gives 98.08% individual latest-value accuracy descriptively. At 16 distinct records, compact COG still trails attention.

## Verification and timings
96/96 checkpoint hashes and recorded evaluations reproduce; NLL error zero. 15552 tokenwise/chunk prediction checks, zero disagreements. 48000 optimizer updates, 1152000 training histories. 57 no-fit geometry/ablation probes and 54 timing rows, not training fits. Warm B1 CPU encoding, prefix state serialization/restoration, four queries; time per correct complete session: narrow SDPA .2783 ms (192/192), full SDPA .2768 (192/192), add128 auto 1.2012 (192/192), delta128 auto 1.5650 (191/192), delta32 16x2 1.5146 (190/192), delta128 direct 1.4078 (191/192). No speed or monetary lifecycle superiority.

## Evidence custody and decision
Full local archive: cog_unified_018_evidence.zip, 25690474 bytes, SHA256 1e0def74f8baeb886de97cc10e0caa71277692939f6eac9f37d54e7f72143025. 96 original optimizer-bearing checkpoints and lossless inference exports with tensor equality checks. Complete raw scores, code, protocols, manifests. All archive members rehashed and ZIP CRC passes. Remote binary/source transfer not yet verified; this record is not weight custody. Prior archive backlog unchanged. No paid compute, GPU, full LM, protected TEST or official campaign credit. Retain associative retrieval as a qualified bounded diagnostic candidate, investigate learned read-only behavior at low state capacity and integration into language-model backbone with competent baselines. Canonical promotion denied.
