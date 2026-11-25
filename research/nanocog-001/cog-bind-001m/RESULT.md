# COG-BIND-001M: learned routing transfers to controlled rewrites, but does not detect missing evidence

## Result
Without retraining, the 001L routed checkpoints retained **6,144/6,144 complete row decisions** on each of the fresh familiar, six-clause neutral-addition and six-clause distractor views. On the view that rewrote the question, declaration and transfer together, **6,050/6,144 (98.5%)** final record/item decisions and **5,892/6,144 (95.9%)** complete stage decisions remained correct. There were **346/384 (90.1%)** all-stage families, but only **114, 127 and 105 out of 128** in the three respective seeds. This is not uniform robustness. Most importantly, when the evidence did not permit a unique answer, the model gave a unique forced choice on every row.

## Frozen and matched design
No model was fitted, tuned, resized in parameter count, or selected using new outcomes. All six original 001L checkpoints were evaluated. Four-clause calls use the exact original forward; larger clause sets use the same weights and equations with the clause dimension generalized. A bounded historical replay matched seven stored raw channels on the first 128 original CAL rows in each checkpoint, and checked identical model outputs at four clauses, plus six-clause permutation and batching checks. This is not full historical-panel replication.

The seven answerable and four unanswerable views share 128 fresh families. Each family has 16 matched presentations crossing alias assignment, transfer direction, query alias and order; all actor atoms were in the original vocabulary. Core transfer pairs and alias–referent associations are absent from actual original study splits. Novel wording introduces new forms and word order with known words, not new vocabulary or authentic free prose. Neutral and distractor views both add two matched six-token clauses at the same positions; only the added content differs. Labels do not enter actor inference. A literal parser verifies support where it is uniquely defined.

## Valid routed results
Each row denominator below is 6,144, and each family denominator is 384 (128 unique families repeated over three seeds). Final accuracy requires correct directed transfer and item. All stages additionally require the correct alias declaration and canonical recipient. Family success requires all 16 presentations.

| Input view | Final row exact | All stages row exact | All-stage families |
| --- | ---: | ---: | ---: |
| Familiar | 6,144 | 6,144 | 384 |
| Only declaration rewritten | 6,144 | 6,028 | 370 |
| Only transfer rewritten | 6,140 | 6,140 | 383 |
| Only question rewritten | 6,040 | 5,958 | 357 |
| All three rewritten | 6,050 | 5,892 | 346 |
| Two neutral additions | 6,144 | 6,144 | 384 |
| Two misleading but irrelevant transfers | 6,144 | 6,144 | 384 |

The bypass solved zero complete families in every valid view. In each six-clause routed view, the paired canonical declaration and transfer, item and recipient all match their familiar baseline decisions on all 6,144 rows, with zero inserted-slot selections. Declaration-only rewrites hid 116 errors in the explicit recipient readout despite perfect final outputs. Question-only rewrites exposed a more consequential first-hop gap: the correct declaration was selected on 5,964/6,144; the item decoder was still perfect when given the correct transfer. On the combined rewrite correct-transfer item accuracy was 6,140/6,144, compared with 6,050/6,144 native final correctness. These stage comparisons localize observed errors but do not prove which hidden representation caused them.

## Unsupported and conflicting evidence
Missing transfer, missing declaration, competing transfers, and conflicting declarations each produced **6,144/6,144** unique model record/item decisions. These outcomes are not ordinary answer accuracy: there is no unique valid target. The architecture has no native abstention action. A descriptive filter requiring each of the bridge, record and item top-score gaps to exceed 2 still accepted **1,862/6,144 (30.3%)** missing-declaration rows and **24/6,144** missing-transfer rows. It retained just **5,442/6,144 (88.6%)** fully rewritten valid rows; all accepted final record/item predictions in that particular valid view were correct. These are selected test diagnostics, not a calibrated threshold or a recommendation for use.

## Verification and next
66 sealed actor-only prediction packets, 135,168 prompt rows, zero training runs. Independent score recount checked the stage, whole-family, and invalid-assignment counts. Keep the learned route as a bounded reference; work on an explicit evidence-sufficiency decision with separate reasons for missing versus contradictory evidence, then validate rejection and valid coverage on new disjoint groups. Separately address question wording that breaks alias selection. No generalized natural-language, safety, hardware timing or attention-replacement qualification.

Data pin `8378ed66cd397b8f37c0fe990db2d4cea3fc0cdaf1d777c589867e69506b15a1`; score SHA-256 `ea53a6e21549d0bf9a573ee7f8cf86a111dde343ed9df1e0dc401fda059bd49d`.
