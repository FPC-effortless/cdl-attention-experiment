# COG-BIND-TIME-RETAIN-012: current turn plus saved state

**The incremental path answered from new input and saved state without rereading any old prompt text.** It matched both history-based modes on all **480/480 turn records per seed**, including cold restores, stale-update rejection, exact source metadata, hard selected positions, and abstentions. All three engineering gates passed, but **only two of three seeds met the complete accuracy gate**: the weakest seed got 45/48 unaffected-board pairs correct (93.75%, below 95%). All modes had those same misses; dropping the historical text did not introduce them. This is a bounded frozen-model prototype, not a qualified general language or lifelong memory system.

At each seed, the comparison processed the same 5,522 new passage tokens and 3,168 new query tokens. It additionally re-encoded **45,266 historical prompt tokens** in the full-history mode or **13,224** in the latest-text baseline; the incremental mode re-encoded **zero**. That is 84% fewer total nonpadding model input tokens than full history and 60% fewer than latest-text in this workload, not zero access to previously learned facts. In the incremental mode those facts were read as selected word IDs and positions in the compact snapshots. The 48 cold snapshot loads consumed 42,754–43,016 compact bytes per seed, with zero restored raw text bytes. Final two-board snapshots were 764–902 bytes per session in this implementation. This is not a claim that state remains that small as the number or complexity of boards grows.

Across the three seeds strict post-restart accuracy was **185/192, 187/192 and 191/192**; strict before-and-after correction pairs were **70/72, 70/72 and 71/72**. The matching misses were abstentions; no emitted wrong answers were observed. Version-valid but ungrounded replacement turns intentionally leave the new version unavailable, instead of re-serving the previous fact. Every stale proposal was rejected before neural passage encoding. Scratch snapshot digests and integer validation can detect corruption in the tested envelope, but this did not test malicious alterations, concurrent writers, process crashes mid-write, or sudden power loss.

**Next:** repair the remaining evidence-acceptance abstentions, then test partial corrections and references such as "change that plan" across many turns and restarts. Continue charging saved state I/O separately from old prompt tokens and compare a quality-matched incremental Transformer/KV implementation before making a KV-cache or production-speed claim.

## Seed-level retained quality and restore

| Seed | Exact asks | After restart | Corrected before→after pairs | Unchanged-board pairs | Stale invariance pairs | Fully exact sessions | Unavailable writes / wrong answers | Disagreements | Gate |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| 20371 | 279/288 (96.88%) | 185/192 (96.35%) | 70/72 (97.22%) | 45/48 (93.75%) | 24/24 (100.00%) | 22/24 (91.67%) | 3/120 (2.50%) / 0/288 (0.00%) | 0/480 (0.00%) | FAIL |
| 20471 | 281/288 (97.57%) | 187/192 (97.40%) | 70/72 (97.22%) | 46/48 (95.83%) | 24/24 (100.00%) | 22/24 (91.67%) | 3/120 (2.50%) / 0/288 (0.00%) | 0/480 (0.00%) | PASS |
| 20571 | 285/288 (98.96%) | 191/192 (99.48%) | 71/72 (98.61%) | 47/48 (97.92%) | 24/24 (100.00%) | 23/24 (95.83%) | 1/120 (0.83%) / 0/288 (0.00%) | 0/480 (0.00%) | PASS |

## Measured input work and persisted state

| Seed | Mode | Old passage parser calls | Old prompt tokens | Old UTF-8 passage bytes | Restoration bytes | Final snapshot bytes / 24 sessions | Cold session-batch elapsed (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| 20371 | history | 984 | 45,266 | 264,336 | 93,868 | 60,909 | 9.288 |
| 20371 | latest | 288 | 13,224 | 77,118 | 55,610 | 27,823 | 8.300 |
| 20371 | incremental | 0 | 0 | 0 | 42,754 | 21,311 | 7.557 |
| 20471 | history | 984 | 45,266 | 264,336 | 93,868 | 60,909 | 8.473 |
| 20471 | latest | 288 | 13,224 | 77,118 | 55,610 | 27,823 | 8.027 |
| 20471 | incremental | 0 | 0 | 0 | 42,880 | 21,305 | 7.111 |
| 20571 | history | 984 | 45,266 | 264,336 | 93,868 | 60,909 | 11.466 |
| 20571 | latest | 288 | 13,224 | 77,118 | 55,610 | 27,823 | 11.157 |
| 20571 | incremental | 0 | 0 | 0 | 43,016 | 21,575 | 10.263 |

Each seed reused the same 24 raw sessions. Counts across seeds are repeated evaluations, not independent sessions. Phase cold elapsed is fixed-order exploratory and includes the child model startup, decoding, per-write persistence and exit; it is not serving evidence. State read bytes are physical snapshot payload bytes read, not just learned input tokens. Input counters record token IDs presented to the neural passage path, and exclude padding and static model/lexicon bytes. The test contains no transformer KV cache in any arm.

## Exact scope

Use the three exact PACKET-009 packet checkpoints and original vocabulary.
No fitting, model selection, or threshold tuning. Raw bounded synthetic
sentences arrive as a single new JSON-line turn through an explicit typed
envelope with board ID and version metadata. The frozen passage branch
proposes/fills packets; LAZY/TUPLE select a coherent hard packet. The
query-only frozen head reads just the question; the exact composer accepts
selected integer raw word IDs and local source coordinates, not the old
passage or a vector cache. This is a scoped prototype and not an LLM.

Compare three paths with identical model weights, inputs and accepted writes:
- History re-encodes all prior version-valid board passages in the current
  session at each question, then uses the requested board's latest packet.
- Latest text retains only each board's latest text and re-encodes just that
  raw passage at a question (stronger same-model baseline).
- Incremental stores both boards' selected token IDs and source positions,
  metadata and source digests, but no raw old passage. It encodes every valid
  newly supplied full-board replacement once and can answer from that state.
All arms run the same neural parser once for every version-valid new
passage so exact write observations can be independently compared. A stale
or invalid version is rejected before model parsing. A legitimate
replacement advances the version and persists its new packet even when
the packet is unavailable; it never silently serves obsolete evidence.
This evaluates full-board replacement, not free-form revision parsing.

The driver alone holds the full synthetic sequence. Each phase launches
fresh model processes. A nested filesystem/PID capsule gives each child
only its exact code/weights/vocab, its own state directory, and one new
turn at a time on stdin. It does not mount the driver input dataset;
in the incremental mode the only previously accepted task content
available is the compact saved state itself. No network-isolation claim.
Each version-valid update persists a per-session snapshot using atomic
rename plus fsync of file and directory. Cold restarts are actual completed
subprocess exits followed by new execs on the same underlying filesystem,
not simulated swaps or power-failure tests. No concurrency,
state authenticity or crash/recovery injection is established.

Report per seed 24 sessions, 20 turns per session, two restart markers
per session, and 12 exact questions per session. Include all transitions
and cold outputs, exact event/item position/word/version, valid and stale
write semantics, mismatch parity across modes, absolute detached truth,
source/version correctness, old passage token/byte reads presented to
the neural parser and physical snapshot read/write bytes. Report failed
or unavailable packets separately; resource parity is not accuracy.

Timing is exploratory fixed-order on one CPU thread per worker; includes
child startup, turn dispatch, parsing, fsync and clean exit in each arm
phase. No competing production transformer, different KV caches, batching
engine or full-model inference baseline is tested. Neither arm implements
an attention KV cache. Reading an explicit token sequence or persisted
state in this benchmark says nothing by itself about reducing KV-cache
traffic in an LLM.
