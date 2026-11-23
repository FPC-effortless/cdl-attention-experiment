# COG-BIND-TIME-REVISE-013: partial edits and retained reference after restart

**Partial edits worked without old prompt reads, but a failed edit can still leave a stale answer.** On all three frozen-model runs, the incremental mode matched the stronger raw-base re-reading comparator on every one of 624 turns. It did not re-encode any old prompt tokens, versus **15,302 per seed** for the comparator; both encoded 2,186 new base tokens and 3,696 new query tokens. The engineered parser processed only the 192 new patch commands per seed, never invoking a neural passage encoder for a patch. Each mode restored focus at all 48 session restore points per seed; the original session script was outside the nested worker mounts.

**Two of three seeds passed the full gate.** Strict question answers were 324/336, 329/336 and 330/336. Exact accepted patches were 67/72, 70/72 and 69/72, against a floor of 69. Of those, 44/48, 46/48 and 46/48 cold "that plan" patches were exact. The weakest seed fell short on exact valid patches, not on mode parity. All 120 ambiguous/unknown/stale or retry controls per seed left state unchanged; the exact negative reason was correct on 118/119/119 because a failed learned base can change the reason to unavailable. No general action-safety guarantee follows.

The important failure was **six wrong emitted answers after legitimate corrections could not be grounded**, shared by both modes. Those answers used the old committed version and returned the wrong item for the intended revision. An exploratory offline check on the same tape marked a version-current unavailable edit as pending on its resolved board and anchor. It would have suppressed all six wrong emitted answers and none of the correct ones on this tape. That rule was **not deployed, did not repair the missing facts, and needs a fresh live test** before claiming improvement. An explicit pending state and expected-version-aware queries are more useful next steps than claiming that a freshness warning alone repairs binding.

The command parser and focus rule are engineered and operate on a declared synthetic grammar, while the frozen neural model supplies base evidence and query positions. This is not unrestricted "change that plan" understanding, a learned language update head, or Transformer prompt/KV-cache elimination.

## Exact held results per seed

| Seed | Exact asks | After cold restart | Registered before→after pairs | Other board unchanged | Other anchor unchanged | Exact accepted patches | Cold “that plan” patches | Unsafe calls blocked | Exact nonmutation reason | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 20371 | 324/336 (96.43%) | 161/168 (95.83%) | 67/72 (93.06%) | 70/72 (97.22%) | 46/48 (95.83%) | 67/72 (93.06%) | 44/48 (91.67%) | 120/120 (100.00%) | 118/120 (98.33%) | FAIL |
| 20471 | 329/336 (97.92%) | 164/168 (97.62%) | 70/72 (97.22%) | 70/72 (97.22%) | 47/48 (97.92%) | 70/72 (97.22%) | 46/48 (95.83%) | 120/120 (100.00%) | 119/120 (99.17%) | PASS |
| 20571 | 330/336 (98.21%) | 165/168 (98.21%) | 69/72 (95.83%) | 71/72 (98.61%) | 47/48 (97.92%) | 69/72 (95.83%) | 46/48 (95.83%) | 120/120 (100.00%) | 119/120 (99.17%) | PASS |

## Mode parity and input cost

| Seed | Mode | Old base prompt tokens | Old raw text UTF-8 bytes parsed | Cold state bytes read | Final snapshot sizes | Exploratory cold elapsed (s) |
|---|---|---:|---:|---:|---|---:|
| 20371 | history | 15,302 | 89,390 | 71,286 | 1405–1727 bytes/session | 9.390 |
| 20371 | incremental | 0 | 0 | 58,688 | 1039–1416 bytes/session | 7.593 |
| 20471 | history | 15,302 | 89,390 | 71,526 | 1551–1727 bytes/session | 8.754 |
| 20471 | incremental | 0 | 0 | 58,922 | 1156–1416 bytes/session | 9.593 |
| 20571 | history | 15,302 | 89,390 | 71,410 | 1435–1727 bytes/session | 10.077 |
| 20571 | incremental | 0 | 0 | 59,076 | 1175–1416 bytes/session | 8.848 |

History here is the stronger latest raw base-text memo, not a whole-transcript replay. Elapsed times are exploratory fixed-order, and capture cold launch, state fsync, raw encoding and answer dispatch. No production-serving or Transformer comparison.

Each seed used the same 24 held sessions; grouping repeated model tests does not multiply the number of independent sessions. Answers require exact version, event, item ID, word and original source position, not merely a plausible string.

Frozen PACKET-009 passage and query heads plus the existing certified
LAZY/TUPLE grouping generate the original selected evidence once at every
initial board write. No new fit or parameter update.

Each correction supplies new text and an expected version, with no typed
board/anchor/operation input. A declared finite grammar parses 'swap/reverse
the order around [anchor] for [reference]'. An explicit 'plan a/b' names its
board; 'that plan' is resolved only by the stored last known-board ASK;
'either plan' or no focus prompts clarification; unknown board, anchor,
failed learned grounding and stale expected versions do not commit.
Vocabulary/grammar rules and version checks are engineered; the experiment
does not claim a learned correction parser or free-form reference resolution.

A valid correction toggles both records' order around exactly one selected
anchor on one board and advances only that board's version. Persist the
latest per-anchor XOR bit and source/version/turn provenance separately
from immutable base packet IDs and source positions. The finite answer
routine views a temporary cue polarity change without altering recorded
source evidence. The item position remains a base raw token position.
A patch requires inferred evidence to support one distinct record on each side of both anchors; writing or asking does not authorize an ungrounded patch.
Only a known-board ask sets or changes focus; this setting is persisted
even if the evidence for the answer is unavailable.

The raw-text baseline retains the latest original base text per board
plus the same typed overlay, and re-encodes only the requested board at
each ask. Its label in native output is 'history'; it is not a deliberately
slow all-conversation replay. The incremental arm stores the exact hard
packet plus compact metadata and focus with no older patch or passage text.
No prior passage is presented to its model during query/patch/restore.
Each nested read-only executable capsule gets only its two sources of
information permitted here: one current streamed message and its own
state files. The raw session script is not in the child mounts. This is
verified for the nested mechanism with inert probes; no network sandbox
claim. The same neural checkpoint loads at each fresh process boundary.

Native deterministic evidence compares version, status, resolved target,
focus, provenance, exact raw witness, all final response fields and stale
or ambiguous non-write paths. Detached truth tests strict exact answers,
registered correction pairs, unaffected boards and unedited anchors,
independently from mode equality. Report old passage calls, raw tokens,
state read/write bytes and mode parity, plus descriptive fixed-order
single-CPU cold elapsed. Neither mode implements a Transformer KV cache.
No free-form editing, crash injection, multiple writers, model-scale
latency or production security guarantee follows.
