# COG-COMPUTE-001 Stage 1f: learned-control adequacy

**Result: 3/3 familiar mastery; 0/3 held qualification.** A stronger generic learned control fit the two-fact task almost perfectly on the maps it trained on, including changing its answer when the question changed. It still struggled when two individually familiar facts appeared together in a new map. No COG variant was fitted in this stage.

## Locked design

The control is a 326,457-parameter fixed-window residual MLP, with 32-dimensional learned token embeddings, width 128, four residual GELU blocks and an ordinary 25-way language head. It physically slices the causal prefix before flattening it into a fixed 14-token grid with zero padding. The same path predicts the next word at training and inference. Eight raw tokens encode two typed triples and a two-token query. Ordinary token CE supervises ANSWER, correct node and EOS; the semantic output is computed before the gold-answer token enters the prefix. No role parser, support labels, explicit comparison, pair loss, persistent memory, candidate mask or fallback. Layout and the fixed-prefix architecture are specialized to this short task; this is not a proposed long-context model or matched-capacity comparison to COG.

Each seed (173, 271, 907) trains from scratch for 4,000 steps, batch 48, 192,000 answer draws, and 576,000 supervised tokens. There are 4,096 distinct training rows across 1,024 labeled fact-pair families. The same six pairs per case per minibatch are sampled with replacement; ordinary CE alone consumes the same raw tokens and labels. Each family tests both questions and both fact orders. The 512 reserved rows comprise 128 exact labeled families absent from training and prior exact panels, with 32 families per case. Individual facts, including their case and physical position, occur in other training maps. Query and target marginals match by case. Under the weaker relation-only renaming check, 0, 4, 4 and 32 of 32 reserved subject-change, subject-invariant, relation-change and relation-invariant families recur in current FIT. No prior CAL family recurs under that rule. This is new combinations of known facts and tokens, not lexical, syntax or abstract-template generalization.

The primary output is unrestricted greedy ANSWER, correct node, EOS. Whole-family success requires all four ordinary outputs right. Both rows and families must exceed 95% in each familiar case and 90% in each held case, separately per seed. A separate public answer marker checks syntax and logits but cannot qualify. All three final checkpoints ran the same original frozen two-part evaluation once, no early stopping or retuning.

## Results

Familiar totals include all four cases. Held change cases listed individually; each of the two held invariant cases was 128/128 rows and 32/32 families for every seed.

| Seed | Familiar rows /4,096 | Familiar complete families /1,024 | Held subject changes answer: rows /128; families /32 | Held relation changes answer: rows /128; families /32 |
|---|---:|---:|---:|---:|
| 173 | 4,096 | 1,024 | 73; 13 | 79; 11 |
| 271 | 4,082 | 1,015 | 83; 16 | 84; 11 |
| 907 | 4,096 | 1,024 | 78; 11 | 85; 15 |

Descriptively pooling the same families over seeds, held answer-changing rows were **482/768** and complete answer-changing families **77/192**. Of the **286** wrong changed rows, **275** chose the other visible fact and **11** chose a value outside the two facts. All 13,824 predictions had valid output format. The public marker logits were bit-identical and rescued none. Familiar and held counts are not independent experimental draws at the row or seed aggregate level.

## Decision

This isolates a capability distinction the small previous baselines did not establish: learning the correct behavior on exact training families is possible under a modest bounded CPU recipe, but no evidence here clears the held generalization gate. The errors favor the competing fact, not formatting or unsupported output as the primary failure. Do not call this a COG improvement, a proof that the task is impossible, or a fair attention-replacement benchmark. Next, a learned comparison control can share one fact encoder across the two input positions and condition on the same query without a supplied correct fact. Compare it to an unstructured control on the same data/exposure and a genuinely new reserved panel. If that works, test whether the useful behavior can be expressed in COG's latent layer rather than adding a persistent store or deeper loops.

## Verification

The first and only native invocation finished 3 FIT and 3 raw OPEN roles in 106.161 serial one-core role seconds (not inference speed). A separate exact recount matched the locked scorer. Aggregate only on GitHub; raw data, checkpoints and replay remain owner-only.
- DATA: `2e8f3f41c0ec17147055373c1b1d5ff61ba46fb1d0b74c1723e7195b4730211c`
- LOCK: `133897588e6c247aae2ef70f707e00e931a56cd58c3c07770967954a0ce666f6`
- WIRE: `ece2b9b622163fcad0475aa9584167914e9c5cff9eef20788e5899d0670dd214`
- SCORE: `3d0476623876d1d007d41b5101dbc5c6f9187b52155706ba625436b8b38969ad`
- Independent receipt: `ac08e4a9f9199e49d9e0874f0fe76c4504dd093ca208bf91c5c1fa4c456f3be8`
