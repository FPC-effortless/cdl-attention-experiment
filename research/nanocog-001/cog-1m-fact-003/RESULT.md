# COG-1M-FACT-003: can a learned subject pointer route a record update?

This is a small two-record mechanism test, not a rerun of the 1M model. A finite grammar exposes raw name, value and disposition spans. A 2,252-parameter model learns the address, commit and value decisions. An explicit two-slot store applies them, and the answer must come from the chosen slot. The literal exact-name parser is an engineered reference, not a competing LLM.

**Native gate: passed.** Learned-advantage-over-initialization gate: passed. Every seed completed at least 127 of 128 four-row families in each held pane, including new spellings, alternate syntax, cancellation, and deliberately swapped write recipients.

| Seed | Familiar strict roots | Novel wording | Unseen names | Recipient-only address swap | Initialized address-only, same learned commit and value |
|---|---:|---:|---:|---:|---:|
| 20371 | 127/128 | 127/128 | 127/128 | 127/128 | address 58/128; unseen 43/128 |
| 20471 | 128/128 | 128/128 | 128/128 | 128/128 | address 61/128; unseen 63/128 |
| 20571 | 128/128 | 128/128 | 128/128 | 128/128 | address 50/128; unseen 40/128 |

Each strict root requires both questions in both worlds to be correct. The answer was constrained by an explicit A/B store API, unlike the 65-way 1M experiment. Incoming names appear verbatim in both mentions, and the parser recognizes two registered finite grammars; this evidence does not qualify aliases, open language, long-context memory, cross-session continuity, or replacement of Transformer attention. Exact literal equality plus the explicit store got all 512 rows per held pane correct, so this is an advantage over the initialized learned router, not over an exact-match algorithm.

With identical frozen weights, forcing only the read address correctly reversed decisions on 508/512, 512/512 and 512/512 address-swap rows. Flipping only the first write address got 127/128, 128/128 and 128/128 decisive rows right before and after intervention, and left the correct answer stable on 383/384, 384/384 and 384/384 cases where that write address should not matter. This tests actual output dependence, not just changes in internal scores.

**Next falsification:** keep the exact weights, expand the candidate set beyond two and add queries whose subject is absent. This study did not test either; a pairwise dot-product router that always selects an available candidate is not sufficient for absence handling. Test known answers, abstention and untouched updates before integrating the router into the larger model.

Official score SHA-256: 67fbc06a856945420ade68d036398fcd5c9afea266c41e979fa1c4f228302656. Independent recount SHA-256: 326e5026de61225ae473292651fbf51a5844078184da156148dfb401dfc6488d. 
