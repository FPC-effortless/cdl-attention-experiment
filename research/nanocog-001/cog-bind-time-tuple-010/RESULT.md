# COG-BIND-TIME-TUPLE-010: coherent clause alternatives with frozen weights

**Outcome: all three frozen checkpoints pass the held-only decoder gate with joint selection, without retraining.** On fresh nested-layout rows, exact native answers rose from 2,372/2,084/2,300 to **3,024/2,984/3,034 of 3,072** (98.44%/97.14%/98.76%). The joint branch preserved every greedy-correct answer and every feasible greedy grouped packet on both held panes. It rescued 652, 900 and 734 previously missed nested cases, with no new wrong commits observed on that panel. This is a bounded synthetic readout result, not retroactive qualification of the PACKET models or evidence of dependable general language, memory or action safety.

**The model's useful evidence was already close to the top.** In the nested test the two-best lists contained all four correct clause tuples in 3,024, 2,984 and 3,040 cases. All available-gold cases were answered exactly in the first two seeds; six available cases were still missed in the third. The remaining errors overwhelmingly came from a needed role not making a top-two clause option. That locates the next candidate improvement in targeted role alternatives and better learned ordering, not a wider blind search or larger memory.

**Accuracy costs work.** The median search examined 8/8/20 cross-record pair-set proposals per novel passage; maxima were 200/256/400, versus a theoretical unfiltered bound of 65,536 four-clause combinations. On the separate plain-CPU DEV timing, joint selection alone had median 237–246 µs per distinct passage, versus 78–141 µs greedy. This excludes the model, composer and all I/O, and neither variant matches the other in quality. It is not a serving-speed or KV-cache result. The next useful comparison is a cheap gate that skips joint search when greedy is structurally feasible. The observed identity theorem supports its correctness on this tape, but this report does not measure that gated variant.

## Enforced native metrics

| Seed | View | Exact greedy | Exact joint | Joint strict phase pairs | Joint anchor flip | Joint anchor invariant | Joint exact quartets | Exact named selected evidence | Joint gate on seed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 20371 | held-familiar | 2,412/3,072 (78.52%) | 2,996/3,072 (97.53%) | 1,498/1,536 (97.53%) | 750/768 (97.66%) | 748/768 (97.40%) | 749/768 (97.53%) | 2,996/3,072 (97.53%) | PASS |
| 20371 | held-novel | 2,372/3,072 (77.21%) | 3,024/3,072 (98.44%) | 1,512/1,536 (98.44%) | 756/768 (98.44%) | 756/768 (98.44%) | 756/768 (98.44%) | 3,024/3,072 (98.44%) | PASS |
| 20471 | held-familiar | 2,188/3,072 (71.22%) | 3,020/3,072 (98.31%) | 1,510/1,536 (98.31%) | 754/768 (98.18%) | 756/768 (98.44%) | 755/768 (98.31%) | 3,020/3,072 (98.31%) | PASS |
| 20471 | held-novel | 2,084/3,072 (67.84%) | 2,984/3,072 (97.14%) | 1,492/1,536 (97.14%) | 746/768 (97.14%) | 746/768 (97.14%) | 746/768 (97.14%) | 2,984/3,072 (97.14%) | PASS |
| 20571 | held-familiar | 2,278/3,072 (74.15%) | 3,060/3,072 (99.61%) | 1,530/1,536 (99.61%) | 762/768 (99.22%) | 764/768 (99.48%) | 763/768 (99.35%) | 3,046/3,072 (99.15%) | PASS |
| 20571 | held-novel | 2,300/3,072 (74.87%) | 3,034/3,072 (98.76%) | 1,517/1,536 (98.76%) | 756/768 (98.44%) | 758/768 (98.70%) | 757/768 (98.57%) | 3,032/3,072 (98.70%) | PASS |

## Decoder diagnosis and paired rescue

| Seed | Complete gold set available, novel | Correctly rescued | Correct cases harmed | New wrong commits from greedy abstention | Novel gain (pp) and paired-root 95% interval | Median / p95 / max checked pair-set proposals per passage |
|---|---:|---:|---:|---:|---|---|
| 20371 | 3,024/3,072 (98.44%) | 652 | 0 | 0 | +21.22 [+16.41, +26.04] | 8 / 64 / 200 |
| 20471 | 2,984/3,072 (97.14%) | 900 | 0 | 0 | +29.30 [+22.14, +37.11] | 8 / 64 / 256 |
| 20571 | 3,040/3,072 (98.96%) | 734 | 0 | 0 | +23.89 [+18.10, +30.01] | 20 / 144 / 400 |

## Which predeclared gates were missed?

- Seed 20371: none.
- Seed 20471: none.
- Seed 20571: none.

## Method

No model was retrained, adapted, or selected. The three PACKET-009 packet checkpoints and their exact vocabulary were reused. Both decoders consumed the same fresh held raw scores from each checkpoint. Each held pane consists of 3,072 cases from 48 new canonical roots, disjoint from every PACKET-009 pane. These are the same known synthetic words and local grammar in new combinations, not a new natural-language source. The novel pane is nested mention order (0110), unseen during the frozen model's fitting; it is not an ordering new to the research program. Familiar and nested panes align on facts, questions and all unordered passage tokens.

The greedy comparison takes the unique best position for each learned role at each detected cue. The joint decoder admits the two best strictly ordered role positions (a second-place tie drops the second position; a first-place tie gives no alternatives), forms up to sixteen distinct-position five-token clauses per cue, and selects one globally consistent four-clause set. Its only structural checks use the raw selected action/item IDs, anchor sets, and nonoverlapping selected intervals; no gold cue, dictionary, boundary, truth, or question is used to search. It chooses the unique maximum sum of the sixteen learned role logits (binary64 accurately rounded sum); a tied complete winner abstains. The original engineered finite composer applies chronology and the question to the selected set afterward.

The search always preserves an already feasible greedy packet on the observed score tape, as verified directly. With strict per-role maxima that packet is the componentwise unique optimum; on arbitrary real-valued input finite-precision score ties remain possible. This preservation property does **not** prevent lower-ranked candidates from converting earlier abstentions into wrong answers. Such commits are counted and are not a general safety test. A correct noun at the wrong occurrence is counted as wrong.

We traced the exact correct five-position tuple availability separately from final selection. The available-gold-but-missed counter can include a different candidate winning, a selection tie, or a downstream failure. It is not a learned retrieval score or a measure of why the error happened. Per-seed relative intervals bootstrap the 48 paired roots, not their correlated rows. Held-only decoder qualification does not retroactively qualify the frozen model's training. The model still uses engineered clause grouping and chronology; this study is not a learned semantic parser, persistent memory, language-model, or efficiency result.

## Compute, provenance, and limits

The raw search bound is 16^4 per passage, but the implementation indexes candidate pairs by selected action/item and anchor keys before comparing groups, and reuses one passage decision across its four questions. Report actual bounded work and verified exact replay; no end-to-end throughput or KV-cache claim follows. Each blind actor ran with its frozen checkpoint and unaided raw inputs in a filesystem/PID namespace; this does not imply network isolation. Exact sealed scores were assessed after an independent release. Scripts, source evidence and conclusions are archived without altering the historical PACKET-009 source.
