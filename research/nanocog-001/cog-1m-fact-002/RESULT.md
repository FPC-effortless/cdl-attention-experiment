# COG-1M-FACT-002: does process supervision improve factual updates?

Process-supervised COG and outcome-only COG began from identical 993,574-parameter initializers and saw the same exact texts, batch order, 576 optimizer updates, and 9,216 answer targets per seed. Process additionally learned the subject's initial position, the effective current value and the accepted/canceled decision from two causal update endpoints. These were 55,296 extra class labels, not extra examples or model parameters, and never supplied at inference. A 1,015,920-parameter full causal attention arm used the same outcome-only supervision and budget.

**No arm qualified.** The registered process-treatment gate did not pass. The added labels helped on the decisive cancellation control: the process model answered **51–59 of 128** changed-target pairs correctly across seeds, versus **0–1** without the auxiliary loss. It did not consistently improve the selected target on unfamiliar wording and worsened the irrelevant-edit control on all three seeds.

| Seed | Arm | TRAIN exact rows / strict roots | Familiar strict roots | Unfamiliar both-world target / strict roots | Cancel both-world target | Irrelevant unchanged / no-op strict |
|---|---|---:|---:|---:|---:|---:|
| 20371 | attention | 768/1536 / 48/384 | 0/128 | 0/128 / 0/128 | 0/128 | 64/128 / 32/128 |
| 20371 | cog-outcome | 1057/1536 / 73/384 | 0/128 | 48/128 / 0/128 | 0/128 | 63/128 / 56/128 |
| 20371 | cog-process | 1156/1536 / 107/384 | 5/128 | 67/128 / 5/128 | 51/128 | 47/128 / 64/128 |
| 20471 | attention | 768/1536 / 48/384 | 0/128 | 0/128 / 0/128 | 0/128 | 64/128 / 32/128 |
| 20471 | cog-outcome | 1066/1536 / 65/384 | 1/128 | 64/128 / 0/128 | 1/128 | 64/128 / 54/128 |
| 20471 | cog-process | 1164/1536 / 113/384 | 7/128 | 66/128 / 8/128 | 57/128 | 50/128 / 64/128 |
| 20571 | attention | 768/1536 / 48/384 | 0/128 | 0/128 / 0/128 | 0/128 | 64/128 / 32/128 |
| 20571 | cog-outcome | 1058/1536 / 62/384 | 0/128 | 58/128 / 1/128 | 0/128 | 54/128 / 55/128 |
| 20571 | cog-process | 1149/1536 / 103/384 | 5/128 | 51/128 / 9/128 | 59/128 | 40/128 / 64/128 |

A strict root requires both raw questions to be answered correctly in both counterfactual worlds. The output is the model’s native unique top character over all 65 symbols, not an A/B-constrained decoder. Training, familiar and unfamiliar examples all use 119 characters and the same Keep/Drop commitment wording. The unfamiliar templates reorder the language; this controls the length and commit lexeme confounds but is not evidence of unrestricted natural language. Held variants share 128 synthetic roots.

Same-weight disabling of the full COG board was a broad diagnostic only. It changes the internal computation distribution, cannot distinguish writing from reading, and is not admission evidence for a particular learned mechanism. No inference-time typed roles, parser, oracle, state serialization, persistent archive or production latency were evaluated.

The fixed-budget attention score is descriptive and an inadequate baseline cannot establish architecture superiority. Model-scope conclusions require learning adequacy before using independent held transfer results. No Shakespeare checkpoint was used for initialization here.

Exact score SHA-256: 2e30db84ebffeb9692f22497f931761314735c4490eb62418d6137acab603c8f. Independent raw recount SHA-256: 24a22b158044e4f5e3bc878de9e3c3bc7a15bd0b19f7f7f9746599e2c253e736.

Next: inspect cancellation handling and independently validate end-to-end competence before scaling delayed tasks or serving tests. The first/second-entity target remains near chance despite perfect local commit classification. The next intervention should verify subject-to-record binding, not simply add more commit labels. A separate admission with new fits and fresh held data would be needed.
