# CSM-SKILL-004 — Transfer a learned choice of controller

Train a depth-three regression tree on measured costs for the primitive, eager SWAP and deferred SWAP controllers over every three- and four-record assignment. At inference, select one controller per goal from its predicted costs, breaking ties in favor of primitive then deferred then eager. The router receives counts of misplaced records, reciprocal pairs and duplicate target values. Freeze training before evaluating every five-record assignment (3,125 goals). No larger roster enters model fitting.

The milestone requires exact final states on every goal, lower total cost than primitive-only planning, and total cost at most equal to the hand-written deferred controller. Compare also to the best single training controller and the per-goal hindsight minimum. Define cost as four units per expanded primitive plus one per controller dispatch. This is a design choice for measuring a trade-off. It excludes routing overhead and does not estimate latency. Publish the outcome even if the learned router misses the milestone.

## Outcome

All arms solved all 3,125 goals. Total costs were 66,365 for primitive, 65,730 for deferred (also the best fixed training controller), 72,590 for eager, 66,180 for learned and 65,690 for the hindsight minimum. The learned choice beat primitive by 185 units and missed deferred by 450. On 150 goals it spent three extra units over deferred and on the rest it tied. It selected primitive on 2,960 goals and deferred on 165. For goal 00043, records 3 and 4 must exchange their values independently of the three slots assigned zero. The learned tree pools this case with other high-mismatch, duplicate-value goals. Its training estimate for deferred ties primitive in that leaf.

The nearest follow-up is to test whether a feature identifying reciprocal pairs whose values are needed exclusively inside the pair resolves this error. Freeze that feature using these diagnostics, then evaluate on a fresh six-record roster. The five-record results are now development evidence. Even a perfect selector has only 40 units of headroom over deferred on this roster.

From this directory use the pinned dependencies in ../requirements.txt and run:

```sh
python experiment.py > /tmp/skill004.csv 2> /tmp/skill004-model.txt
diff -u scores.csv /tmp/skill004.csv
diff -u model.txt /tmp/skill004-model.txt
```
