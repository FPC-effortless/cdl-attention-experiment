# COG-1M-FACT-001: matched short-context factual binding

This study trained the original 993,574-parameter COG architecture at a fixed budget to read synthetic two-entity histories and predict the current value for the queried entity. It compares COG and a 1,015,920-parameter full causal attention model from scratch; a third COG arm reuses the exact prior Shakespeare checkpoint as a separate transfer test. All runs use unconstrained 65-character output, not forced A/B ranking.

**No arm qualified.** No registered architecture advantage was established. The Shakespeare warm start raised complete familiar cases to 31–42 of 128 per seed (scratch COG: 3–13), but ended at 0 of 128 on new wording for every seed (scratch: 1, 4 and 1). None of the nine fits reached the prespecified training competence floor.

| Seed | Arm | TRAIN exact rows / strict roots | Familiar strict roots | New-wording edited both worlds | New-wording strict roots | Irrelevant-edit target | No-op strict roots |
|---|---|---:|---:|---:|---:|---:|---:|
| 20371 | attention | 768/1536 / 0/384 | 0/128 | 0/128 | 0/128 | 64/128 | 32/128 |
| 20371 | cog-scratch | 1160/1536 / 51/384 | 11/128 | 31/128 | 1/128 | 62/128 | 64/128 |
| 20371 | cog-warm | 1308/1536 / 195/384 | 31/128 | 3/128 | 0/128 | 62/128 | 70/128 |
| 20471 | attention | 1152/1536 / 0/384 | 0/128 | 0/128 | 0/128 | 64/128 | 64/128 |
| 20471 | cog-scratch | 1155/1536 / 4/384 | 3/128 | 55/128 | 4/128 | 65/128 | 64/128 |
| 20471 | cog-warm | 1325/1536 / 218/384 | 42/128 | 4/128 | 0/128 | 72/128 | 71/128 |
| 20571 | attention | 1152/1536 / 0/384 | 0/128 | 32/128 | 0/128 | 61/128 | 64/128 |
| 20571 | cog-scratch | 1183/1536 / 64/384 | 13/128 | 43/128 | 1/128 | 65/128 | 66/128 |
| 20571 | cog-warm | 1339/1536 / 223/384 | 38/128 | 0/128 | 0/128 | 69/128 | 73/128 |

**How to read this.** A strict root includes both worlds and both raw questions. The main target requires a correct factual change in both worlds; the untouched entity, the irrelevant edit, and a commutative no-op also have exactness gates. The held novel pane changes template and length jointly (82–86 familiar characters versus 125–126 novel). Held panes share 128 semantic roots and are not independent examples.

The matched workload was 384 CPU optimizer updates and 6,144 answer targets per fit, without Shakespeare rehearsal, extra role labels or held tuning. Execution adequacy verifies the budget and input custody, not semantic competence. These models are parameter-near, not FLOP-matched. The study does not test restart, learned archive writes, general-language facts or a cost advantage at equal accuracy.

A next diagnostic can distinguish basic optimization from transfer: require both scratch arms to clear a predeclared training and development competence floor before comparing on a separate fresh held panel. Use same-length held rephrasings with a fixed random pairing and separately cross length with wording so a failure can be attributed. These length and positional diagnoses have not been run. Keep the warm-start result separate from the architectural comparison.

Exact score SHA-256: 30bfba5ead3e8203036992c7930f65842beb9b7e22a4e9edc91632c76eb9f9de. Independent recount SHA-256: 166203da7e5b6bae2f265683ecc98aed4c7d439efbd186b00d8850ca76626221.
