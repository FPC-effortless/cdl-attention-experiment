# COG-COMM-001: data and engineering preflight

A memory-free comparison is implemented in nanoCOG: learned local composition, exclusive prefix summaries, bounded learned routing, and a content-blind subscriber with the same scratch geometry. Here “memory-free” means no writable semantic store or cross-document memory; prefix and packet state still use storage within a prompt.

## What has run

The independently reviewed Forge DATA-only materialization prepared one fixed, constructed corpus, with disjoint aliases between TRAIN, CAL and DEV. The panels use the same grammar.

| Role | Families | Worlds | Encoded tokens/world (range) | p95 |
| --- | ---: | ---: | ---: | ---: |
| TRAIN | 320 | 640 | 162–440 | 406 |
| CAL | 120 | 240 | 161–441 | 402 |
| DEV | 240 | 480 | 159–449 | 406 |

All 240 DEV families have the same query and exact local suffix across their two opposing worlds, with both source color tokens outside the local decoder’s eight-token field but inside the 64×8-block age rule. A deterministic grammar reference recovered both answers in all 240 pairs; each of five simple answer shortcuts recovered zero pairs. These checks establish a usable constructed test, **not** a model score, absence of all possible shortcuts, or transfer beyond this grammar. Forge reported no findings for the exact content/group split checks it ran. Some diagnostic families were not run or do not apply to the label-free Forge projection.

A narrower 16-block age rule allowed direct support in only 16/80 inert fixture worlds. Widening the age envelope does not address earlier eviction: the current router retains at most eight payloads per layer and stream, and supplies no more than four candidate slots at a read. “Age eligible” does not establish publication, retention, discovery or causal use.

The engineering suite exercises future-token noninterference, chunk parity, reset/fork isolation, actor projection, paired scoring and strictly past route positions. It is separate from scientific qualification.

## Decision

**Training and prediction remain on hold.** A/B/C are nested configurations, not capacity-matched isolated mechanisms. The fixed-subscriber control is available with matched parameter registration but does not train the subscriber. Before committing the main fit budget, demonstrate an easy local learnability control with matched loss and verify a real support-survival denominator across the per-token write schedule. Freeze a real-shape CPU cap. The inert eight-world, 448-position learned-route probe took about 0.81 seconds for forward and backward versus about 0.035 seconds for the summary-only reference, so the prototype has not earned a speed claim. No model was fit, no model predictions on the corpus were run, and no protected data or paid resources were used.
