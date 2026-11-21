# COG-LM-003 | Can longer training and forced routing teach revision?

**Decision. Keep the bank experimental.** A continuous 2,048-token delay restored a gradient to the earlier fact. An explicit utilization penalty produced much larger reads. Neither intervention taught this bank to answer counterbalanced reworded corrections. All ten fits failed the promotion criterion.

## Attribution of the original failure

On original 002 prompts we intervened on the exact successor prediction while preserving the trained sequence model, learned bank and prefix. At 256 intervening tokens, full COG got 14/16 literal answers and suppressing the successor contribution gave 0/16. At 1,024, both conditions scored 0/16. The original 256-token success depended on literal retrieval. Those scores alone cannot identify a recurrent-state decay time. The 002 utilization figure is a median norm ratio, not a fraction of gradients, operations or correct answers.

## What was trained

Three jointly pretrained 002 sequence processors and their tied embeddings were frozen. The COG revision variants share the same 31,014,179-parameter parent, including the previously trained bank. COG and Mamba lookup-only models start from their respective matched 002 parents. Every fit updates the existing language adapter and lookup mixture. Bank fits also update content keys, queries, write strengths, salience, values and readout. All non-masked tests use the ordinary generation path with lookup and local recurrence available.

We built 48 distinct training episodes from eight pairs of names, three filler delays (64, 512 and 2,048) and both orientations of each assignment. Each episode gives two people the same initial direction, changes the first person’s direction and then queries either the revised or unaffected person. Wording includes literal and reworded questions, and three correction phrasings. The 24 paired filler selections come from the WikiText-2 training split. Pair members have identical filler and question text and opposite targets. At test, four new pairs of names, a new correction phrasing, two new query forms and held-out Wikipedia filler test generalization together.

Each fit used eight epochs, 192 AdamW updates, 384 question presentations and 353,888 input tokens counting repeated presentations. Only eight filler next-token positions per presentation were supervised (3,072 presentations of natural-text targets). Revision-supervised fits also predicted 384 final answer tokens with full-vocabulary NLL, never the gold slot ID or a gold plan. Five primary COG-bank and four lookup-control fits share a seed and data order. The tenth fit is the post-hoc amplitude-penalty experiment. This is a small, deliberately targeted adaptation experiment. It does not represent 353,888 densely supervised language targets.

The 128 condition resets the sequence model, bank and exact table every 128 input tokens. The continuous condition retains all three. Comparing long and short therefore tests the horizon of the entire system. In masked training, 192 questions restart local recurrence and exact lookup at question onset, with the bank carried across that boundary. Direct answer gradients can still reward a context-independent word preference. The regularized run adds a one-sided squared log shortfall relative to 1% read amplitude at training answer positions; its term has weight 0.25. The term does not supervise which fact is written or read.

## Delayed retrieval and natural language

| Training condition | Reworded top-1 at 1,024 | Both assignment directions preferred | Median read / hidden at that query | WikiText NLL ↓ | Dialogue NLL ↓ |
| --- | --- | --- | --- | --- | --- |
| 128 reset, text targets | 0/16 | 0/8 | 0.0307% | 5.0682 | 7.2800 |
| Continuous, text targets | 0/16 | 0/8 | 0.0002% | 5.0188 | 7.2144 |
| 128 reset, revision supervision | 0/16 | 0/8 | 0.0175% | 4.9785 | 7.0708 |
| Continuous, revision supervision | 0/16 | 0/8 | 0.1722% | 5.0029 | 7.0899 |
| Continuous, local path masked in training | 0/16 | 0/8 | 0.0453% | 5.0345 | 7.1015 |
| Masking + read-amplitude penalty (exploratory) | 0/16 | 0/8 | 42.6769% | 5.0788 | 6.9773 |
| COG lookup, intact training | 0/16 | 0/8 | 0.0000% | 4.9762 | 7.0592 |
| COG lookup, masking | 0/16 | 0/8 | 0.0000% | 5.0193 | 7.0675 |
| Mamba lookup, intact training | 0/16 | 0/8 | 0.0000% | 4.6147 | 7.0522 |
| Mamba lookup, masking | 0/16 | 0/8 | 0.0000% | 4.7020 | 7.1754 |

All ten fits scored 0/16 full-vocabulary reworded answers at **each of 256, 1,024, 4,096 and 16,384 intervening tokens**. For the primary revision-trained COG model, 4/16 literal responses were correct at each distance. The fresh task differs from 002, so its literal counts should be compared within 003. Across the eight counterfactual pairs per delay, both answer directions must be preferred to count as content-sensitive retrieval. A fixed direction preference scores 8/16 individual forced choices here. The 16 trials reuse four name pairs and one seed; the four horizons are repeated measures, not independent replications.

On intact training-episode replay the pressure-trained COG, the floor-trained COG, the long-supervision COG and both pressure-trained lookup controls each gave 48/96 correct top-1 answers to familiar reworded questions, yet none preferred the correct answer in both orientations of any of the 48 pairs. This catches a shortcut that would look like partial binding in ordinary accuracy. The paired test already fails on the familiar examples.

The standard WikiText and MultiWOZ next-token panels use 15,616 and 12,288 targets. All 003 variants lost natural-text prediction quality relative to the relevant 002 parent. The original jointly trained COG revision checkpoint scored 3.9299 WikiText NLL. The pressure-trained revision head scored 5.0345. Repeated sparse filler targets and highly concentrated answer supervision limit inference about better-balanced training.

## Gradient and bank-use measurements

To isolate the bank path, we differentiated the final correct-minus-foil log probability with respect to frozen backbone representations at the initial fact. Four paired training streams used a 2,048-token delay. For the 002 bank, the median prefix-feature gradient norm was zero with 128 resets and 4.14e-07 with a continuous bank. After route-masked training it was 9.63e-06 with continuity. The exploratory penalty raised it to 8.42e-05. These are local derivatives conditional on the four sampled streams and checkpoint, not gradients through trainable backbone parameters.

| Bank configuration | Natural-text median read / hidden | Natural-text 95th percentile |
| --- | --- | --- |
| Original 002 checkpoint | 0.0111% | 0.038% |
| 128 reset, text targets | 0.7914% | 39.864% |
| Continuous, text targets | 0.8201% | 34.460% |
| 128 reset, revision supervision | 0.0021% | 0.037% |
| Continuous, revision supervision | 0.0843% | 0.654% |
| Masking + read-amplitude penalty (exploratory) | 7.1237% | 47.194% |
| Continuous, local path masked in training | 0.0013% | 0.035% |

The panel is the first 1,024 tokens of each of 58 eligible WikiText articles. On held-out reworded queries the exploratory penalty increased the read amplitude enough that suppressing the read changed 34 of 64 top predictions. No unmasked reworded prediction was correct. Larger activations and a trainable path are insufficient evidence of a useful representation.

## Authentic changes of mind

| Condition | Correct switches on known test / 137 | Correct switches on fresh dev / 156 | Fresh exact whole answers |
| --- | --- | --- | --- |
| parent_cog_revision | 19 | 26 | 0 |
| 128 reset, revision supervision | 22 | 23 | 0 |
| Continuous, revision supervision | 20 | 23 | 0 |
| Continuous, local path masked in training | 25 | 28 | 0 |
| route_revision_off | 25 | 28 | 0 |
| Masking + read-amplitude penalty (exploratory) | 22 | 29 | 0 |
| floor_revision_off | 23 | 30 | 0 |
| COG lookup, masking | 25 | 25 | 0 |
| Mamba lookup, masking | 20 | 21 | 0 |

Both panels use actual MultiWOZ transcripts with an explicit correction and an evaluator-supplied question. The transcript and question are model inputs. The two old/new strings are used only for scoring. A correct switch prefers the old answer before the change and the new one afterward. The fresh shard contains 139 distinct dialogues. The two-candidate most-recent-mention rule gets 122/137 and 147/156 switches. This rule benefits from access to the two candidates, whereas exact generation is full-vocabulary.

The primary bank and its cost-matched masked read have identical switch indicators in every case on both panels. Against the trained COG lookup control the bank has three additional fresh switches, but the gain survives turning the bank off. The paired, dialogue-cluster bootstrap for the difference versus Mamba is stored in evidence/summary.json. Equal switch counts on their own would not prove identical behavior.

## CPU cost

| Model at normal inference | Cached decode tokens/s | Encode + prefill ms | Warm request ms |
| --- | --- | --- | --- |
| Unadapted 002 Mamba + lookup | 155.9 | 564.0 | 775.0 |
| COG lookup, masking | 187.8 | 345.8 | 524.6 |
| Mamba lookup, masking | 147.9 | 529.0 | 734.0 |
| Continuous, local path masked in training | 163.8 | 376.9 | 579.6 |

Eight rotated and reverse-order warm trials per model, two CPU threads, batch one, full FP32, a 1,024-token prompt and 32 greedy steps. Timing includes memory maintenance and output decoding. Prompts, generated tokens and individual timing records ship with the release. CPU throughput is specific to this host. No matched attention model was run.

## Interpretation

The intervention establishes that a 128-token reset blocks the bank’s cross-boundary credit path, while continuous computation permits it. There was already an opportunity to learn writes and reads within 128 tokens. Across matched short and long supervision, simply restoring the long path gave no paired recall gain. Training the model to use a larger memory signal did not teach it which entity owned which updated value. These results leave open the roles of curriculum size, targeted objective, optimization and the representational limits of the existing bank.

The next decision should turn on **paired content sensitivity at short delay**. Check whether a matched model can reverse its answer for an unseen assignment of two familiar entities with the same surrounding text before paying for a longer context or more parameters. A larger set of independent bindings, a counterfactual training loss and broad natural-text rehearsal are specific candidates for that next test.

## Verification

The sparse training implementation matched the native 002 readout on continuous and 128-reset paths to within 3.82e-6 in log probabilities and 4.77e-7 in gradients. All ten trained heads reloaded with matching hashes. Native end-to-end logits reproduced cached counterfactual odds on the checked held-out prompts (maximum absolute difference 1.91e-06). The CPU checks for 001/002 also passed. README.md gives reproduction commands and the archive includes raw scores, training curves and exact checkpoints.
