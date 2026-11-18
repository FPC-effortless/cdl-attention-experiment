# CSM-SKILL-005 — Recognizing independent SWAP pairs

Freeze two depth-three cost-regression trees fitted only to the 283 three- and four-record assignments from Study 004. Both use the same targets, tie order and training algorithm. The treatment adds the number of reciprocal pairs for which each source value occurs exactly once in the goal. Five-record results informed the feature. Six-record assignments are reserved for a single exhaustive evaluation after the trees and protocol are committed.

Primary gate: solve all 46,656 six-record assignments with cleared scratch and total learned cost at or below the fixed deferred controller. Cost is four units per expanded primitive and one per dispatch, excluding routing overhead. Compare with the original feature set, primitive and eager controllers, and per-goal hindsight minimum. Publish all rows and any missed gate.

## Outcome

The feature and both trees were frozen in commit 4d1f743. Every controller reached the exact goal on all 46,656 assignments. The independent-pair selector matched deferred cost on each individual goal. It selected primitive 42,981 times and deferred 3,675 times. The preregistered gate passed.

| Controller or selector | Total weighted cost |
| --- | ---: |
| Primitive | 1,219,765 |
| Deferred | 1,210,120 |
| Eager | 1,319,320 |
| Original three-feature tree | 1,219,570 |
| Independent-pair tree | 1,210,120 |
| Per-goal hindsight minimum | 1,209,685 |

Adding the engineered feature saved 9,450 units against the matched learned control. An always-deferred controller already achieves the same total without a router. The 435-unit gap to hindsight bounds the remaining opportunity under this objective and these three controllers. The measurement covers one acquired procedure and a structured assignment family. A useful next test is whether selection transfers across two acquired procedures in a task family where the best fixed controller leaves a substantial gap to hindsight.

Replay from this directory with Python 3.12 and the dependencies in ../requirements.txt.

```sh
python study.py fit > /tmp/skill005-models.json
diff -u models.json /tmp/skill005-models.json
python study.py evaluate > /tmp/skill005-scores.csv
diff -u scores.csv /tmp/skill005-scores.csv
```
