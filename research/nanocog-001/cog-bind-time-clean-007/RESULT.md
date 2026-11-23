# COG-BIND-TIME-CLEAN-007: temporal routing through a hard role interface

**Outcome: PASS on the registered bounded gate (3/3 seeds pass all requirements).**

## Trained outcomes

All numbers below are exact counts from sealed full actor outputs. Strict pairs count only when both answers identify the correct source and item position. Quartets require all four same-passage answers. Seeds are repeated fits on the same data, not additional independent test examples.

| View | Seed | Exact event and item | Phase pairs | Anchor-flip pairs | Anchor-invariant pairs | Question quartets | Exact requested witness | All four relations and query | Exact query | Abstentions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dev | 20371 | 1,536/1,536 (100.00%) | 768/768 (100.00%) | 384/384 (100.00%) | 384/384 (100.00%) | 384/384 (100.00%) | 1,536/1,536 (100.00%) | 1,536/1,536 (100.00%) | 1,536/1,536 (100.00%) | 0/1,536 (0.00%) |
| dev | 20471 | 1,536/1,536 (100.00%) | 768/768 (100.00%) | 384/384 (100.00%) | 384/384 (100.00%) | 384/384 (100.00%) | 1,536/1,536 (100.00%) | 1,536/1,536 (100.00%) | 1,536/1,536 (100.00%) | 0/1,536 (0.00%) |
| dev | 20571 | 1,536/1,536 (100.00%) | 768/768 (100.00%) | 384/384 (100.00%) | 384/384 (100.00%) | 384/384 (100.00%) | 1,536/1,536 (100.00%) | 1,536/1,536 (100.00%) | 1,536/1,536 (100.00%) | 0/1,536 (0.00%) |
| held-familiar | 20371 | 3,072/3,072 (100.00%) | 1,536/1,536 (100.00%) | 768/768 (100.00%) | 768/768 (100.00%) | 768/768 (100.00%) | 3,072/3,072 (100.00%) | 3,072/3,072 (100.00%) | 3,072/3,072 (100.00%) | 0/3,072 (0.00%) |
| held-familiar | 20471 | 3,072/3,072 (100.00%) | 1,536/1,536 (100.00%) | 768/768 (100.00%) | 768/768 (100.00%) | 768/768 (100.00%) | 3,072/3,072 (100.00%) | 3,072/3,072 (100.00%) | 3,072/3,072 (100.00%) | 0/3,072 (0.00%) |
| held-familiar | 20571 | 3,072/3,072 (100.00%) | 1,536/1,536 (100.00%) | 768/768 (100.00%) | 768/768 (100.00%) | 768/768 (100.00%) | 3,072/3,072 (100.00%) | 3,072/3,072 (100.00%) | 3,072/3,072 (100.00%) | 0/3,072 (0.00%) |
| held-novel | 20371 | 3,072/3,072 (100.00%) | 1,536/1,536 (100.00%) | 768/768 (100.00%) | 768/768 (100.00%) | 768/768 (100.00%) | 3,072/3,072 (100.00%) | 3,072/3,072 (100.00%) | 3,072/3,072 (100.00%) | 0/3,072 (0.00%) |
| held-novel | 20471 | 3,072/3,072 (100.00%) | 1,536/1,536 (100.00%) | 768/768 (100.00%) | 768/768 (100.00%) | 768/768 (100.00%) | 3,072/3,072 (100.00%) | 3,072/3,072 (100.00%) | 3,072/3,072 (100.00%) | 0/3,072 (0.00%) |
| held-novel | 20571 | 3,072/3,072 (100.00%) | 1,536/1,536 (100.00%) | 768/768 (100.00%) | 768/768 (100.00%) | 768/768 (100.00%) | 3,072/3,072 (100.00%) | 3,072/3,072 (100.00%) | 3,072/3,072 (100.00%) | 0/3,072 (0.00%) |

### TRAIN competency and registered failures

| Seed | Exact TRAIN route | All 22 locations correct | Novel role-swap reversals among already correct and witnessed rows | Failed requirements |
|---:|---:|---:|---:|---|
| 20371 | 6,144/6,144 (100.00%) | 6,144/6,144 (100.00%) | 3,072/3,072 (100.00%) | none |
| 20471 | 6,144/6,144 (100.00%) | 6,144/6,144 (100.00%) | 3,072/3,072 (100.00%) | none |
| 20571 | 6,144/6,144 (100.00%) | 6,144/6,144 (100.00%) | 3,072/3,072 (100.00%) | none |

Registered floors: ≥95% novel exact rows, ≥90% novel strict phase, anchor-flip and anchor-invariant pairs, ≥85% novel complete quartets, ≥95% novel requested witness, ≥95% familiar exact rows, and ≥95% in both TRAIN audits, separately for every seed.

## Engineering record

The first launch stopped before any model worker or optimizer step when the dispatcher tried to parse a manifested non-JSON input as JSON. The dispatcher now validates bytes independently of format. The original attempt and lock were retained, and the revised prelaunch audit reached the worker boundary for all three cases while executing none. The runtime change affects only the wrapper; the model, corpus, seeds, objective, gates and scorer were unchanged.

## What changed

COMP-006 removed the old event head, but its named role positions still carried contextual vectors with information from the whole event. A learned composer could therefore pass by ignoring the intended role distinction. CLEAN-007 removes that path: a new learned locator sees raw words and predicts five role positions per candidate relation and two query positions. The output composer sees only the chosen raw token IDs and their integer positions. It cannot see token embeddings, pooled vectors, confidence scores, old model outputs or gold inputs. There is no pre-existing model, hidden answer lookup or fallback scorer. Modifying non-selected words with the hard packet held fixed cannot change the composer, but it can still change the learned locator when the locator is rerun.

The composer is deliberately **engineered**, not learned. It matches a selected query anchor to a selected relation in each of two records. Given a recognized temporal cue, it calculates phase using whether the trigger or focal action is the first selected operand after the cue. Exactly one record must satisfy the selected query phase. The returned answer is the selected raw item **at the correct position in the correct record**. A repeated noun in the wrong record does not count. Invalid or ambiguous choices abstain and are scored wrong.

The fitted locator has 19,399 trainable parameters. At each of three fixed seeds, AdamW performs 384 updates on 6,144 distinct training rows in 96 factorial roots, yielding 49,152 repeated row exposures, 2,457,600 input-token exposures, and 1,081,344 explicit location-label exposures. Only these supervised location targets enter the objective, not a target event/answer loss. Record and relation delimiters are given in the observed text. Each held view has 3,072 rows from 48 disjoint roots, and the two aligned views have identical passage token bags and question text for each matched row. Individual tokens and local frames are familiar; the harder view holds out combinations of the frames inside records. Rows and the two aligned views are not statistically independent.

## Negative controls and scope

Both operand types start with the same determiner, which breaks the old cue-plus-next-word shortcut (768/3,072 correct even with the requested anchor supplied; 1,536 abstentions). On each pane, an anchor-ignoring decision rule that otherwise knows the phases has a ceiling of 75% exact rows and 50% exact same-passage question quartets. On the same text, reversing the requested phase always changes the correct record; changing the requested anchor switches the record in half the cases and leaves it unchanged in the other half. The four-question quartet demands all decisions simultaneously. An exact scorer-only resolver verified the fixture truth; it is not used during model inference.

Exchanging the selected action and trigger **positions** or flipping the selected cue's polarity applies the same algebraic involution to the fixed composer. These interventions verify the declared rule, not two independent discoveries by the neural model. Exact learned witness scores also verify the selected query fields and both requested relations against the labels, so a correct final answer alone is not sufficient. The results cannot establish outcome-only binding, general-language transfer, free generation, end-to-end efficiency, or a better attention replacement. There is no matched architecture comparison. The test is a bounded, directly supervised learning and composition study.

FIT payloads include TRAIN only. In the blind actor, a read-only filesystem allowlist and isolated PID namespace exclude the truth files; the actor writes predictions to a separate output location. This is not network isolation. The model-free scoring process requires separately admitted immutable ledgers and scorer digest before reading truth. The trained actors were not used for development-time selection or threshold adjustment.

## What the result changes

The corrected interface passed the full finite test. Every seed selected all 22 annotated token positions correctly on both held panes, and the same-passage answers changed or stayed fixed as required. This is stronger evidence for learned local role grounding than COMP-006's event score: the finite composer cannot receive a whole-event hidden vector under a misleading role name, and both the selected witnesses and exact answer provenance were correct. It is not a matched quantitative win over COMP-006, which used different tasks, supervision, and readout.

The result resolves the tested ambiguity under explicit span supervision and known language primitives. It does not resolve learning how to ground unfamiliar constructions with fewer labels, or learning the composition rule itself. The most informative next comparison is to keep this passing hybrid control frozen, build a fresh test with varied operand placement, implicit clause boundaries and independently sourced wording, and compare the same hard interface with reduced or no per-role supervision. Report exact roles and exact answers separately before considering scale.
