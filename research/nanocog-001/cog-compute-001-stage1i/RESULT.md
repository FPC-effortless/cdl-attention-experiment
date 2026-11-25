# COG-COMPUTE-001 Stage 1i: Slot-relative match and payload separation

**Negative result.** Splitting what a span uses to match the query from the value it passes on did not repair raw-token binding. The split reduced exact changed-answer predictions from **535/1,152 to 507/1,152**. In the coherent value-reassignment diagnostic it reduced cases where both assignments were answered correctly from **73/1,152 to 28/1,152**. Neither arm completed an eight-example reassignment family or passed either held panel in any seed.

## Frozen comparison

The baseline is the byte-identical Stage 1h sender-conditioned mixer, freshly fit. Its projected three-token span feeds both the content gate and the value projection. The candidate uses the same 67,847 parameters and same tensors at initialization, but the shared sender affine is factored across input columns: the first two relative token slots feed the gate's sender path, and the third feeds the payload path. A shared bias and separate GELU transforms are used. Both use the same causal receiver, fixed lag code, channel gates, structural averaging, and full-vocabulary decoder. Fabricated tests verify per-span isolation and exact zero cross-path gradient under the structural split. The same token can still affect another overlapping span in a different role. This tests a joint positional bias, not two separately identified effects or a general-purpose parser.

Both score all eligible previous contiguous trigrams, including overlapping windows. Each sees only 12 raw prompt tokens, with two intact facts among three nuisance tokens, and a visible query delimiter. Neither model is told which windows are facts, which one is correct, or which answer tokens are eligible. No external state, memory bank, support label, oracle gate, loop, or checkpoint transfer was used. Both trained with the same stratified draws and ordinary next-token cross-entropy, 4,000 updates × 48 rows per fit, three fixed same-init seeds. Same parameter count and exposure do not imply identical computational cost.

4,096 training prompts; 768 fresh primary prompts over 96 semantic maps (24 per case), both queries and both fact orders, rendered once each in familiar versus unseen gap configurations. These are the same underlying maps with matched fact order, query and filler across panels. Primary and supplemental labeled maps are absent the tracked earlier splits and the current FIT; current held key pairs are absent FIT and Stage 1h's held skeletons. Each individual held literal fact is present in the same case and both record ranks in FIT. Established vocabulary, grammar, and single fact assignments are not new language. The supplementary 384 prompts swap the two literal values in the changed-answer worlds with both keys, query, layout, and noise fixed. Every affected window sees the coherent raw edit. One all-eight family requires both assignments, both queries and both orders, not merely changing the guessed token.

## Results

| Exact greedy outcome | Full-span interaction | Factorized |
|---|---:|---:|
| Familiar passes across all case floors | 2/3 seeds | 1/3 seeds |
| Primary known-layout passes | 0/3 seeds | 0/3 seeds |
| Primary unseen-layout passes | 0/3 seeds | 0/3 seeds |
| Known-layout changed rows | 274/576 | 258/576 |
| Unseen-layout changed rows | 261/576 | 249/576 |
| Known-layout full changed families | 15/144 | 12/144 |
| Unseen-layout full changed families | 14/144 | 9/144 |
| Both original and reassigned query answers exact | 73/1,152 | 28/1,152 |
| Same output node despite changed correct answer | 991/1,152 | 1,099/1,152 |
| All eight reassignment examples correct | 0/288 | 0/288 |

Held pass requires strictly above 90% exact rows and complete four-world families within every case for that seed. These descriptive totals repeat the same semantic cases across seeds and layouts. All 31,488 observed generations emitted the public answer marker; scoring with a supplied marker matched the natural answer-site logits exactly. Thus output formatting is not an explanation. The split rescued 162 changed-answer predictions the control missed but lost 190 the control got right. It made fewer guesses outside the two visible values (22 vs 60) and more wrong-visible-rival guesses (623 vs 557). The equal-value control improved (1,148 vs 1,133 correct of 1,152), but it does not require selecting the right fact.

## Interpretation

Do not promote this factorization or argue that a narrower value channel alone learns binding. Its imposed separation works at the level of one span, but the output follows the current assignment less reliably. The prior frozen gate intervention showed that selective routing can help given existing activations; it did not show that this positional split supplies that selectivity. The new primary maps differ from Stage 1h, so Stage 1h's historical scores are not the control here. A next bounded test should isolate local composition before routing rather than scale this split or introduce persistent memory. Retain the coherent reassignment check and explicit, equally treated baselines.

## Reproducibility

Forge actor-only diagnostics, six FIT and six raw-only OPEN roles in TrainLab, original detached scoring and independent raw recount. The one CPU invocation used 169.936615 summed role seconds, not serving evidence. Gate, native receipt, and archive integrity do not themselves qualify model competence. DATA `ffba1a0c5eb277317f0065d0831da9bb3a6255938f4822b1e551bec1dbf8d001`; LOCK `a2f7c6ba01c05439b96717a4833fcc3e52bb8191dcb226c760648da259be90be`; WIRE `8273dc414319d1cdfcd49a50b4d30546919bbe14eed8faea1ea0564065554dc3`; score `8d3af83d4955372da52d7ee42b6a467516262aaedd568417ca2a65ad1d818be8`; independent receipt `8ba955f6fcd54e7a4f146174a672e770ec1b1fc8cab897a175c5e58c46a7dee7`. [Original Stage 1h comparator](https://github.com/FPC-effortless/cdl-attention-experiment/blob/aa2e97ac03edf1ae9500fab8caed61d42ab9b05a/research/nanocog-001/cog-compute-001-stage1h/RESULT.md) · [Frozen localization](https://github.com/FPC-effortless/cdl-attention-experiment/blob/1debb99a6fc862739f6d4c7282eb3e443e09684a/research/nanocog-001/cog-compute-001-stage1h-localization/RESULT.md).
