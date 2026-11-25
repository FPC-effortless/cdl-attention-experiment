# COG-COMPUTE-001 Stage 1g: shared learned fact comparison

**Result: the shared-fact control passed on all three seeds; the flat control passed on none.** On held combinations where changing the question changes the correct answer, the shared model completed 192/192 four-view families, versus 80/192 for the flat model. Both models had mastered familiar cases and received identical raw input, supervision and draws. This qualifies a task-shaped learned positive control, not COG or a general attention replacement.

## Exact comparison

Six fresh fits: seeds 173, 271, and 907 for the unchanged fixed-window PrefixMLP (326,457 parameters) and the new shared-fact control (326,313). The candidate uses public fixed offsets for two subject/relation/value triples and one subject/relation query. It separately projects the query and each subject/relation key, applies the same learned scoring function to both pairs, normalizes the two scalar scores, and mixes shared projected values. A learned decoder reads the mixed latent value and actual causal output prefix and predicts over all 25 tokens. No symbolic equality, provided support record, key lookup, persistent state, answer copy, mask, or fallback. This is bounded two-item attention-like selection with fixed typed segmentation. Fact-order symmetry is imposed by sharing and symmetric aggregation; it is not a learned invariant attributed to the data. Several architectural features change together; this does not identify the isolated contribution of encoder sharing.

Both arms see identical 4,096 Stage 1f TRAIN rows, paired sample sequence, 4,000 updates ×48, and ordinary next-token CE on ANSWER, node and EOS. Both use raw causal prefixes; no future answer is visible at the semantic prediction. Flat refits reproduce the previous final-state hashes. Same parameters to within 144 and exact data exposure do not imply the same training FLOPs. The new held panel has 512 prompts in 128 exact labeled families, balanced across subject/relation changes and whether the answer changes, both orders and both queries. Each held single fact appeared in another training family in the same case and both physical positions. No exact labeled family overlaps train or previous studies, and no prior reserved family recurs under relation-only relabeling. Under that weaker equivalence active TRAIN shares a homologue with 0/4/3/32 reserved families, respectively, of 32 in the four cases. This is recombination of familiar facts, known symbols and fixed grammar, not novel syntax, records of unknown format, or abstract topology.

One final checkpoint per fit, full-vocabulary free greedy ANSWER, correct node, EOS; no answer constraints. Strict gate >95% familiar and >90% held on both rows and complete families within each case for every seed. The separate supplied-answer-marker probe checks logits and cannot qualify the model.

## Native held performance

Each row entry is exact native rows /128 and complete four-view families /32. All invariant cases for all six endpoints were 128/128 rows and 32/32 families.

| Architecture | Seed | Subject changes answer | Relation changes answer |
|---|---:|---:|---:|
| Flat | 173 | 76; 11 | 85; 16 |
| Flat | 271 | 80; 11 | 83; 18 |
| Flat | 907 | 75; 12 | 83; 12 |
| Shared fact | 173 | 128; 32 | 128; 32 |
| Shared fact | 271 | 128; 32 | 128; 32 |
| Shared fact | 907 | 128; 32 | 128; 32 |

- Flat: 3/3 cleared all familiar strata, 0/3 cleared all held strata. Shared: 4,096/4,096 familiar rows and 512/512 held rows for each seed; 3/3 passed both gates.
- Across seeds, total held rows: flat 1,250/1,536; shared 1,536/1,536. The shared model corrected all 286 flat misses without harming any held row.
- Among answer-changing pairs, both queries were correct in 180/384 flat and 384/384 shared. Entire changed families: 80/192 and 192/192. Repeated families across seeds and layouts are descriptive counts, not independent draws.
- Swapping the whole fact order gives the shared candidate bit-identical fixed-marker logits in every scored panel/case, consistent with its built-in symmetry. The model still had to learn how the query should change the output; the order symmetry alone cannot select the right fact.

## Decision and verification

This is a reproducible learned solution for the specific typed two-fact setting, and a competent reference for the next question: can COG learn an equivalent pre-pooling comparison without being given the fact boundaries? Keep the qualified reference intact. Compare raw-only alternatives as layout, placement and distractors vary, and separate boundary discovery, fact selection and answer decoding. Do not claim improved COG, attention-free status for the control, scalable general-language reasoning, or an inference win.

One original native invocation finished six FITs and six OPENs in 208.442 serial one-core role seconds (not serving latency). Separate independent exact recount agreed with the original score. Only aggregate evidence is public; synthetic data and six new checkpoints are owner-only.
- DATA: `ea2bf84160fae31f8dc9e530d92ab1c291ed503201b57deba2fc28a0ed13f4cf`
- LOCK: `6588eaca249bd7f7e9587baf12b83a3562e0d4a5edabfec76c283c9d3724fd3e`
- WIRE: `84d14630cb6e32da19958b8ac2e6944a92688ef96c6031289d72e267a275fa24`
- SCORE: `34bfbb95bb620f29a14b54935575b69e988d560e81e5f4867b9ba40a0cc00806`
- Independent receipt: `bd3092b5359a6f2857a7e9726a1e57f45e2145ce985fb847e46c7e2465b2d551`
