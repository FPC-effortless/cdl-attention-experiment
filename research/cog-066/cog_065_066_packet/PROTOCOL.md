# COG-ATTN-066: procedural chunks versus exact primitive search

Pre-fit protocol. Local synthetic parcels only. No real consignments, external messages, paid compute, protected TEST, or publication. Do not acquire, time, or score these panels before independent review of exact FREEZE.json.

## Question and prior
PR #72's 051–060 packet used template rebinding for listener accuracy (052), mined subpaths/reverse-policy reuse (055), and supplied task IDs for the retention task (057). Veritas NX-018 previously found mined macros underperformed primitive search on held-out and fresh panels. This is not a claim to invent macro mining. Here the separate test is verified execution under recombination and observed mid-execution drift with the same transition and search opportunity. Short sequences are acquired from verified primitive traces across different source families, with no supplied macro labels or boundaries and no task identifier at held planning (training family identity is provenance metadata for counting distinct traces only). Are these chunks useful when combined in new workflows, at the same quality and lower complete cost than a tuned exact primitive planner? This is two-level symbolic reuse, not broad neural skill learning or agent qualification.

## Input and policies
Three simulated items each have three single-use stages at four bays. A primitive moves or advances one authorized item at its current bay. Movement costs 1+distance and work costs 1. Goal is each requested stage count. Exact primitive A* uses an admissible lower bound (mandatory work plus distance to cover all required bays), best-state pruning, skips moves to bays with no ready work (intermediate empty trips are dominated by direct travel), deep-first tie-breaking, and a 4,096-pop aggregate budget per case. An independent verifier recomputes actual transitions, revisions, authority and costs from receipts. A visible version change aborts the old plan and replans from actual observed state. No simulated effects are counted as executed.

Candidate: top 16 length 2–4 action chunks mined with frequency-compression proxy and minimum eligibility of eight occurrences across at least three successful TRAIN families. Only verified primitive traces drive acquisition. Keep all primitive edges; options only add valid multi-step edges. Test exact primitive search; candidate; fixed 16-entry designer dictionary; and three seeded within-option-order permutations (6501–6503). Disclose duplicates/effective dictionary size and that fixed lengths need not match learned lengths. Every plan expansion and primitive transition attempt counts; macro processing is not treated as free.

All planners are uncached across calls and revisions: each starts a fresh queue and best-cost map. The mined, fixed, and shuffled dictionaries persist across calls but store no task plans, compiled transitions, or verified outputs. Applicability scans, rechecks, and version-triggered replans count. No inference about memoized variants.

## Data and phases
Enumerate 8^3 route signatures, assign via committed SHA256 modulo 10. TRAIN buckets 0–5; fresh development panels 6–7 and 8–9. All worlds share constituents, but no signature or generated item name crosses panel boundaries. The two panels are independently named held splits of the same recipe distribution, not two task distributions. Names are nonfunctional for planner; no identity-generalization hypothesis. Each signature creates identical goals across success, initial hold, and mid-plan visible revision variants. Data generation is deterministic and hashed before fitting. No prior TEST set is consulted.

1. Seal exact sources, metadata and runtime using seal only. Invariant unittest uses handwritten examples and split metadata; no held execution.
2. Independent technical approval of exact FREEZE.json hash. Only then run acquire on TRAIN.
3. Seal full acquisition traces, per-primitive receipts, and independent unpruned Dijkstra cost replay before predict. Predict freezes every arm on both dev panels and a fixed timing subset, then saves SHA256. Each timed call's output must equal the corresponding saved deterministic prediction, checked outside the timing window.
4. Verify independently checks receipts, static authorized primitive-reference minima against an unpruned Dijkstra oracle, and complete result inventory; saves results and gate. No data or code edits after evaluation; fresh future panels for redesigned arms.

## Metrics and gates
No quality regressions: statuses match reference per case, no increased primitive cost on reference successes, and independent receipt verification proves no stale effects in replay. Record observed and incomplete paths; retain negative results. Report true cost/all transitions and exploratory controls; repeated timings are not independent statistical samples.

Timing: deterministic hash-selected whole families, all three conditions; warm starts then seven interleaved measurement blocks. Each timed call includes plan, any replan, execution checks, independent verification and the identical receipt text template. The template is a correctness/rendering control here, no training claim. Measure the candidate acquisition + construction cost (primitive baseline needs neither; cannot use log training-cost ratio against zero). For 512 operational workflows and 4,096 sensitivity, compare T_build + 12 * N * I. Charge shared expenses equally; record runtime/version and one-host limits. Independent oracle audit overhead is logged separately from deployment trace acquisition and chunk build. A positive paired difference means time saved. Require candidate ≥10% median family saving against A*, lower bound of paired family-bootstrap mean ns saved >0, mean saving no more than 2% of baseline median call worse than fixed dictionary, and lifecycle delta <0 at N=512, in addition to quality parity. No claim beyond synthetic fixture workload even on a pass.

## Run (pinned canonical ecosystem; scientific inner kernels use stdlib)
From the packet parent, run invariant-only tests, then make the science seal and prepare the native Forge metadata/custody freeze. No acquisition, prediction/timing, or scoring is allowed before independent review of the exact pair of freeze hashes.

- python3 -m unittest -v cog_065_066_packet.test_fixtures
- python3 -m cog_065_066_packet.run seal
- python3 -m cog_065_066_packet.native prepare
- After separate exact approved stage review: python3 -m cog_065_066_packet.native acquire
- Then separate approved predict and verify calls with their own predecessor receipts.

The native runner uses the previously recovered and verified Veritas ecosystem at eec439492407b0aa26922c23b5ae40d95289f4eb and 062's isolated exact compatible packages. It does not mutate the prior study or install dependencies. Forge audits metadata only. TRAIN and DEV inputs are staged separately for TrainLab subprocesses; only learned option closure, not raw training traces, enters prediction. Each recipe validates scope, dictionary cap, fixed resource bounds and appropriate output inventory. Measurements use normal uninstrumented subprocess execution and the same timed worker for every arm. Canonical Veritas evidence and Research Lab bindings are diagnostic only, NOT_QUALIFIED. See NATIVE_WORKFLOW.md.
