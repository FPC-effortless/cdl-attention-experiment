# Independent learned archive review

## Scope

Read-only review of the experimental, host-trusted `learned_archive.py`, its
prospective protocol, data generator and locked fit/evaluation runner. Review
focused on stale writes/reads, retries, erasure, immutable acceptance authority,
model identity/hooks, and separation of labels from inference. No production
source was edited by this reviewer. The model-path owner made the two repairs.

The deterministic reproduction fixture implements only the native model's
interface and uses hand-set output heads. It tests infrastructure behavior and
provides **no learned semantic accuracy evidence**.

## Reproduced defects and repairs

| Defect | Reproduction | Repair and verified result |
|---|---|---|
| Global PyTorch inference hooks bypassed the attachment guard | Register a global forward hook that changes the actual factual writer's output from the Amara operation to Bela while all tensor bytes remain unchanged. The old `observe()` accepted it. | Reject global forward and pre-forward hooks in the same guard that checks local hooks and model identity. Independent test now rejects the call. |
| Writer revision captured after prediction could rebase an older in-flight event | During prediction for an Amara write, a second store inserts or updates the target to Bela. The old code then captured Bela's revision and overwrote it with Amara. Both missing-key and existing-key cases reproduced. | Read the bounded eight-key revision set before prediction, then bind the chosen operation to its original key revision. Both interleavings now raise a conflict and retain Bela. |

The original implementation has SHA256
`49b5063d9add7ccebdd91dc0509e41994285c86cee8037d37fac65d17ad57aba`, matching the
initial data manifest. The fixes had already landed when the reproduction copy
was taken, so the original was reconstructed by reversing only these two changes;
its full-file hash was verified against that manifest before running the tests.
The fixed source has SHA256
`25c043c7f83f12e38f506064cd79ebd85914227ac2b5ca7b9ab61acc6178034b`, matching the
subsequent calibration/source lock.

- Original source: **3 tests failed**, reproducing both defects.
- Fixed source: **3 tests passed**.
- Exact test source hash, source hashes, exit codes and complete pytest outputs
  are retained in `reproduction.json`, `before-pytest.txt` and `after-pytest.txt`.

The writer guarantee begins at each key's snapshot. Reading eight keys separately
does not provide a total event-time ordering across concurrent conversations;
only the selected record is written, with compare-and-swap against its captured
revision. Query reads similarly need not form one multi-key read transaction:
the answer's final transaction revalidates **every actual read key** and live
source before committing. An insertion, correction, erasure or recreation after
any read changes its revision or invalidates its source and blocks the stale
answer. Stable absent/deleted states remain distinguishable by revision.

## Other reviewed contracts

The decision receipt persists its operation, base revision, model identity and
authority before proposal acceptance. A retry resumes that same proposal and
cannot silently adopt a newer fact revision. Acceptance replay returns its
historical accepted revision without reapplying it; changing authority is
rejected. Ask replay intentionally returns a historical exact-request receipt,
while a new request reads current facts. Source dependencies run from observation
to decision to accepted fact and answer, supporting transitive erasure.

Tensor-byte changes before or during inference are checked. Local and global
inference hooks are rejected after the repair. Loader-admitted host code and
class implementations remain a trust boundary; this review does not claim a
sandbox against arbitrary Python code changes.

No train/test label leakage was found in the reviewed fit/evaluation flow.
TRAIN feature extraction uses TRAIN text; held-out text features are extracted
after fitting with the unchanged frozen native model. Model prediction receives
text features and its typed archive read value, never gold query routes, ledgers
or answers. Gold labels are confined to loss construction and scoring. Separate
view RNG preserves matched semantic family/world/event sampling across training
arms. Feature caches and exported predictions were subsequently checked against
their extraction, native-model and checkpoint hashes before diagnostic replay.

## Separate frozen-prediction diagnostics

The locked primary scorer omitted the protocol's commitment/address/value and
required-chain breakdown. `diagnostics.py` fills that reporting gap by replaying
the six **already frozen prediction exports**, without model calls, fitting,
checkpoint selection, oracle correction or primary endpoint changes.

Every original field for **6,912 pair rows** reconciles exactly, including
operations, ledgers, routes, answers, first error and damage. All familiar TEST
pairs remain correct in every arm/seed. Every arm/seed still scores **0/256**
strict fresh pairs. The candidate's required-chain counts are 9/256, 0/256 and
2/256 across seeds17/29/41. Seed17 has nine pairs with both answers right, but
only **eight** satisfy both the required-chain and answer conditions.

The candidate misses fewer required writes on fresh wording, but makes more
unrequested writes and joint address/value mistakes. Damage to previously
correct unchanged records rises from 108/145/16 to 698/834/811; these are raw
counts with different opportunities to damage a previously correct record, not
exposure-matched rates. Initial fact establishment is already fragile: the first
event is wrong in 386/464/366 of512 candidate histories. All typed proposals in
these fresh panels are valid, so schema rejection does not explain the failure.

These families shift both task vocabulary and syntax. The result cannot isolate
word-order or binding generalization from unfamiliar lexical associations, and
one frozen native backbone with three adapter-head seeds cannot establish a
general claim about COG. The complete tables, definitions, denominators, subgroup
counts and limits are in `DIAGNOSTICS.md` and `diagnostic-summary.json`.
