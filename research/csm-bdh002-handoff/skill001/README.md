# CSM-SKILL-001 — Skill acquisition with procedural memory

## Research direction

Developmental Artificial Cognitive Architecture is a proposed skill-first direction for CSM. Make acquisition of reliable internal operations an explicit training objective. Language can express, teach, and coordinate acquired skills. Demonstrations, interaction, and state outcomes offer independent evidence that the system can perform the operation.

Start with recognition, association, discrimination, binding and recollection. Build toward composition, abstraction, prediction, planning, verification, adaptation and transfer. A skill `πₖ(S, g) → S′` is a parameterized state transformation under an objective and a contract. A complex skill is a composition `πₙ ∘ … ∘ π₁`. Admit a candidate to the procedural library after checking its correctness, transfer and cost on independent tasks. Retain explicit preconditions, effects and reusable arguments. Track discovery, verification, retention and reuse over successive acquisitions.

| Component | Proposed role |
| --- | --- |
| CSM | Execute typed operations and preserve working state |
| LRC | Retrieve relevant skills and compose relationships |
| BDH-inspired memory | Acquire temporary associations rapidly |
| State-space mechanisms | Process continuous sequences |
| World model | Predict consequences of candidate operations |
| Lean verifier | Prove specified procedure properties against a formal execution model |
| Procedural memory | Store, retrieve and invoke admitted procedures |
| Executive controller | Select, sequence and verify skill applications |

Relevant precedents include library induction in DreamCoder, temporally extended actions in hierarchical reinforcement learning, compositional meta-learning and learning as program induction. CSM can test which of these ideas help under persistent relational memory, typed execution and explicit verification. [DreamCoder](https://arxiv.org/abs/2006.08381) · [Options framework](https://doi.org/10.1016/S0004-3702(99)00052-1) · [Compositional meta-learning](https://doi.org/10.1038/s41586-023-06668-3) · [Learning as program induction](https://doi.org/10.1126/science.aab3050)

## Curriculum and comparisons

1. Learn elementary state operations such as SET, COPY, LINK, DELETE and QUERY.
2. Discover short procedures from goals and demonstrations.
3. Consolidate procedures as parameterized skills and verify their contracts.
4. Retrieve and compose skills on unseen task combinations.
5. Add conditional execution and iteration.
6. Test unfamiliar task families, delayed retention and independent transfer.
7. Measure acquisition from demonstrations without weight updates, including whether later skills require fewer demonstrations or less compute.
8. Add linguistic names and descriptions after skills have executable semantics.

Compare outcome-only, operation-demonstration and procedural-memory learners. Demonstration arms should receive identical traces and all arms should share objectives, initial states, executor and charged compute. Track acquisition speed, independently held-out transfer, retention, composition depth, demonstration count, inference latency and total cost, including compilation and verification. Measure the 12:1 inference-first target only after correctness qualification. A planning work count can help explain a latency change, while the target requires an end-to-end timing comparison at matched correctness.

## Pilot

`pilot.py` uses a finite symbolic state machine with COPY and scratch DELETE. It infers a four-operation SWAP sequence from one state-transition demonstration, then binds its abstract operands to concrete records. Admission checks 162 combinations of values and addresses, including equal input values, an unrelated record and cleared scratch. An incomplete procedure fails admission. This is executable finite checking.

All arms use breadth-first search under identical primitive-operation budgets. The flat demonstration arm learns action-type ordering from the same trace as the procedural-memory arm. The memory arm additionally instantiates and tries admitted macros. Each macro candidate is charged for every contained primitive. The outcome-only arm induces a seed solution from its initial and goal states. Subsequent goals expose only the initial and desired state. The transfer set consists of two goal structures across 24 assignments of unseen values, with each goal expressible as a composition of two swaps.

| Primitive budget per task | Outcome-only | Flat demonstrations | Procedural memory |
| ---: | ---: | ---: | ---: |
| 512 | 0/48 | 0/48 | 48/48 |
| 2,048 | 0/48 | 0/48 | 48/48 |
| 8,192 | 0/48 | 0/48 | 48/48 |
| 32,768 | 48/48 | 48/48 | 48/48 |

At the budget where all arms succeed, the memory arm was charged 3,120 primitive checks versus 842,184 for flat demonstrations and 1,048,824 for outcome-only search. These totals cover search. Admission and demonstration processing are outside the transfer count. The test family was selected to exercise learned SWAP composition. Breadth-first search is a weak planning baseline for permutations, and the 48 cases are correlated value renamings of two goal structures. A goal-directed planner and tasks that require discovering which prior skill helps are needed to isolate a wider acquisition advantage. This pilot has no Lean proof, timed inference result, retention study or evidence of meta-learning.

Replay with the Python standard library from this directory:

```sh
python pilot.py > /tmp/csm-skill001.json
diff -u pilot-results.json /tmp/csm-skill001.json
```
