# COG-COMPUTE-001 Stage 1c: query-conditioned local composition

**Result: 0/9 qualified.** When the two facts implied the same answer, all three architectures got every test item right. When changing the question required a different answer, individual accuracy stayed near picking one of the two facts without using the question. The receiver-conditioned COG produced two complete relation-switch families in one seed and none in the other two. This does not settle whether a stronger learned model or a different objective can learn the task.

## Exact design

Three seeds (173, 271, 907) each used fresh width-48 COG, receiver-conditioned COG and tied causal attention. All apply a single shared block twice and use the same sinusoidal absolute position features and output convention. Parameter counts were 25,056, 25,460 and 24,432; candidate adds 404, with the subset shared by both COGs initialized identically. The original uses one learned scalar gate per causal offset; the candidate adds a rank-4 receiver-conditioned term with separate mixing by channel. Sources remain the same five fixed lags; logits at a fixed pass do not inspect the sender. Both COGs retain the existing causal summary path. No persistent state, symbolic parser, externally supplied location, pairwise attention in COG or extra passes. This comparison changes parameter count and kernel work modestly and is not exact FLOP matching.

Two fact triples plus subject and relation query give eight raw tokens. Both facts can influence the actual answer choice after the generated ANSWER marker via two radius-four passes, even with the summary path disabled in the fabricated check. Changing only query subject or relation changes candidate marker gates on the second pass in the fabricated test. These checks establish availability, not useful semantic use. There are 2,304 distinct training examples in 576 labeled families, with exactly 6,000 draws per case per fit, raw-prompt and final-answer supervision only, and 1,000 updates of batch 24. The equal-value/different-relation training case deliberately repeats only 12 distinct families. Cases and raw inputs are matched across all arms, with no output-choice masking. No held test outcomes contributed to training or stopping.

OPEN comprised 384 raw examples, 96 disjoint exact labeled fact pairs, with each family crossing both fact orders and both queries. This tests transfer to fresh combinations of familiar symbols and four fixed templates, not unfamiliar syntax, entities, or graph topologies. Each of the four cases contains 24 families, 96 rows and 48 same-facts query pairs. Query, target-slot and answer marginals are balanced across cases; answer-changing pairs change exactly one raw query token and preserve both facts and their order. Each case required at least 87/96 exact free outputs **and** 22/24 fully correct four-world families. Each seed and each case had to qualify.

## Changed-answer results

Cells are exact free rows /96 and fully correct families /24.

| Seed | Arm | Subject changes answer | Relation changes answer |
|---|---|---:|---:|
| 173 | COG | 46; 0 | 48; 0 |
| 173 | Receiver-conditioned COG | 49; 0 | 49; 0 |
| 173 | Tied attention | 52; 0 | 48; 0 |
| 271 | COG | 48; 0 | 48; 0 |
| 271 | Receiver-conditioned COG | 49; 0 | 50; 2 |
| 271 | Tied attention | 56; 4 | 48; 0 |
| 907 | COG | 48; 0 | 47; 0 |
| 907 | Receiver-conditioned COG | 48; 0 | 48; 0 |
| 907 | Tied attention | 55; 1 | 48; 0 |

Among all 288 answer-changing query pairs per architecture across seeds and axes, original COG answered both correctly in 4, receiver-conditioned COG in 9, and tied attention in 26. These counts include both layouts and repeated test families across seeds, so they are descriptive. Every arm and seed was 96/96 exact and 24/24 complete on **each** invariant case. All 3,456 outputs across both types were correctly formatted. Supplying the public answer marker was bit-exact with the ordinary next-token prediction on all rows and rescued none. In invariant cases an answer can be obtained without deciding which fact the query names; perfect invariant performance is not proof of binding.

## Decision and verification

The tested receiver modulation did not solve two-fact selection. Weak performance from the small tied attention control as well means the result does not isolate a defect unique to COG and does not qualify a fair efficiency comparison. Do not add memory or multi-hop passes to fix this result. A focused next test is to pair the two differing questions in the **training objective**, forcing a preference reversal over the same fact list, against unchanged ordinary answer training with exactly the same input draws. Start with these local conditions, use separate fresh combinations, and keep test outputs unassisted. That would distinguish an objective shortcut from a limitation that survives paired training.

One native invocation: nine fits and nine raw evaluations, 98.3931 serial role seconds on one CPU core (not inference speed). Independent exact recount matched. Public aggregate; private replay carries original synthetic outputs and fits.
- DATA: `7560ee147f84a2e0a25d84b027501acdc4c0f5fb3e194c158bd903d20146a5d5`
- LOCK: `9dac2a211e10167e91fb3de7af7f71a11680c064c2115cee04bfa7c385478e5b`
- WIRE: `05e2385da6c92bf7dfffdca132755d0639c2e9f6e0792d9ddc0e801c7cb3d8a6`
- SCORE: `d95180563fc51bff9f106f138a25f9395abf4f3c9f8c0af9eeb200f40d1cd946`
- Independent receipt: `f75b1a1be62f2754f043aac164ca0d9ee727fae597bfeab0c46deda1a12df40f`
