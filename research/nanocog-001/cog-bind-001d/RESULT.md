# COG-BIND-001D: a reader repair recovers known rewrites but barely transfers
## Decision
The correct value was accessible to a new readout of frozen 001B record
vectors for much of the specific shifted wording that broke 001C. For fresh
groups rendered in that same shifted style, giving the correct record yielded
455/1,536 strict families with the frozen original reader, 458/1,536 when
refitting the reader on familiar wording alone, and 1,214/1,536 when refitting
with matched familiar and shifted wording (29.6%, 29.8%, 79.0%).
This establishes recoverability for the trained style on this task. It is not
a general syntax fix: on the two new wordings the matched mixed reader only
reached 58/1,536, versus 46/1,536 before (3.8% vs 3.0%).

## Matched control
Six 001B checkpoints, original selector and representation frozen. The exact
48-dimensional value projection is passed to each reader with no text, query,
slot position or true record as a feature. Each new reader has 2,744 scalars
and starts from the exact same original weights. Over 2,932 paired unique TRAIN
clauses, the familiar reader repeats the familiar form; the mixed reader sees
one familiar and one shifted form of the same record. AdamW, 1,840 paired
updates and 117,280 slot exposures per head. Neither CAL nor HELD was used to
choose weights. All readers score all six records; the known-source diagnostic
uses the true record only in scoring. Native joint also requires the exact
unique support selected by the unchanged ranker. Both metrics demand the
unique correct color on all 16 target rows per group.

256 fresh 2×2 held rectangles exclude all 256 previous 001B/001C rectangles
and exact prompts, but share all 144 individual entity–relation keys with
the old held split. Three versions keep facts and the familiar question fixed.
NF means two new fact templates made from known tokens, absent from all model
and readout training. This is not generalization to unseen keys or natural text.

| Wording | Known source: original / familiar / mixed | Native joint: original / familiar / mixed | Exact support ceiling |
| --- | ---: | ---: | ---: |
| Familiar | 1534 / 1536 / 1502 | 1366 / 1368 / 1335 | 1368 |
| Shifted, taught to mixed reader | 455 / 458 / 1214 | 264 / 266 / 691 | 882 |
| New templates, never taught | 46 / 50 / 58 | 21 / 25 / 35 | 850 |

Each entry is out of 1,536 model–family evaluations; seeds share test groups,
so these are descriptive counts, not independent replications of each item.
On the taught shift, mixed versus familiar repair rescues 779 and loses 23
known-source families; it rescues 441 and loses 16 native joint families.
The known-source gain occurs in each of the six checkpoints. Familiar
performance is slightly worse under mixed repair (34 known-source losses
vs the familiar refit, zero rescues), so preservation is not perfect.
For the new templates the mixed reader's small known-source gain is mixed
across checkpoints. A post-hoc row breakdown finds the fronted-value form
much harder (2,204/12,000 for mixed, vs 11,510/12,576 in the other new form);
this compares templates, not word position in isolation.

## Interpretation and next
Start with reader and selection robustness, not bigger memory. Reader repair
alone raises native joint completion on the taught shift only to 691/1,536
(45.0%); the frozen selector independently caps this condition at 882/1,536.
The next candidate should train a wording-robust record encoder and selector
with matched controls, then evaluate fully new paraphrase families and report
both native and known-source completion. The current new templates are now
an exposed diagnostic, not valid for further qualification. A weak result with
a particular decoder does not prove the frozen vectors lack the value.
No new parent fit, language-model, speed, multi-hop, plasticity or system pass.

## Verification
Six matched reader pairs and 18 sealed prediction lanes (92,160 prompt rows).
The frozen hook reproduced the native decoder logits exactly on every forward
batch. Ranking is generated once from the frozen parent for all three readers.
The score uses exact support/answer maxima with tie rejection, and retains
all raw ranks and all per-slot answer logits. No deployment.
