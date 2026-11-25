# COG-PULSE-004A-R: memory influenced answers, but the reader did not qualify

The reader answered some matched counterfactuals correctly, but no seed met the predetermined competence gate. Simply putting the right fact into memory did not make the model reliably bind the query to that fact or prefer it over an older revision.

| Intervention | What happened across three seeds |
|---|---|
| Correct fact supplied | **7, 6, 6 of 24** complete case families; **108, 106, 100 of 192** individual answers correct. |
| Current fact replaced with its counterfactual | The same **7, 6, 6 of 24** families were fully correct both before and after replacement. Each family crossed both presentation orders and both current/unrelated values. |
| Relevant evidence removed | **0/24** complete factual families per seed. Only **0, 4, 4 of 192** outputs emitted the explicit no-evidence token; failing to answer correctly was not reliable abstention. |
| Unrelated fact replaced | The output stayed unchanged on just **98, 86, 96 of 192** queries, including wrong answers that stayed wrong. It stayed unchanged and correct on **90, 86, 84 of 192**. |
| Obsolete revision present alongside current fact | **0/24** complete families per seed; **19, 42, 25 of 192** current answers correct. Most failures were no-evidence outputs, not direct copies of the obsolete value. |
| Native memory disabled | **0/24** complete factual families per seed. The no-evidence token appeared on **184, 168, 192 of 192** outputs. |

The TRAIN foothold also failed: **0, 2, 1 of 72** complete known families, against the registered floor of 65. On the companion test where only the raw question switched to the other owner, both answers were correct in **3, 13, 6 of 192** matched examples, never across a complete case family. These are results for this reader and budget, not against a competing architecture.

The pilot used the unchanged native 48-way PULSE Publisher reader, initialized fresh for three 480-update CPU fits. The open constructed intervention harness knew which record to remove or swap, and supplied only observed record tokens, the unmodified raw question, and occupancy to the model. The six treatments did not pass answer targets into the native head or payload. Slot order and the visible revision ordinal were varied independently; no privileged priority marked the newer record. This was not a protected held-out evaluation or an autonomous learned write test.

**Next:** first qualify query-to-record selection on the same supplied physical bank, with each record scored against the query before mixing. Then test visible revision ranking within the selected identity while retaining absence and irrelevant-edit controls. Don't tune the write objective around this reader until those tests pass. The current pilot does not show that the bank mechanism uniquely caused the failure, that training longer would fix it, or that PULSE replaces attention.

All six original native roles succeeded. An outer checker initially stopped because it expected the wrong text label on a valid execution receipt. The separately reviewed continuation reused the first fit and ran only the five previously unused roles. No fit was repeated. Exact aggregate score SHA-256: `5e2f531e360df729ae0e1b952aa4c64099fa61e38d75ff975d8f53e89c8cc9c3`. Exact open output pin: `8b093dba5d1a62df6c550455680e0769f19e17ecf781608669358b33221275b9`. Independent recount SHA-256: `16b42ab0dfc277e02f21511a9d5891248321f6897ae6f035b8d7c1bc0ce9053b`.
