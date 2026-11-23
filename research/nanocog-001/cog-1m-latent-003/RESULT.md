# COG-1M-LATENT-003: the prefix board, not the retained controller, recovered the lost answers

**Both frozen checkpoints passed on fresh cases.** The fixed 224-byte handoff matched the full literal path: **36/36** and **35/36** exact answer pairs. No optimizer ran. On the weaker checkpoint, retaining the prefix board recovered two fresh pairs and nine previously lost pairs; retaining the controller alone recovered none.

| Complete exact answer pairs | Neither retained | Controller only | Board only | Both retained |
| --- | ---: | ---: | ---: | ---: |
| Fresh, checkpoint 20371, compact or literal | 36/36 | 36/36 | 36/36 | 36/36 |
| Fresh, checkpoint 20471, compact or literal | 33/36 | 33/36 | 35/36 | 35/36 |
| Previously exposed, checkpoint 20371, compact or literal | 63/64 | 63/64 | 63/64 | 63/64 |
| Previously exposed, checkpoint 20471, literal | 54/64 | 54/64 | 63/64 | 63/64 |
| Previously exposed, checkpoint 20471, compact | 54/64 | 54/64 | 64/64 | 63/64 |

The exposed replay used the original LATENT-002 cases and reproduced the original packet and reference generation tapes bit for bit. Keeping both components in the compact path recovered exactly the same nine pairs as the literal path. The extra exposed win from compact board-only was one pair; adding the controller removed it. The fresh set reproduced the board effect on the weaker checkpoint, not that extra win. The prior 64 pairs are not part of the fresh score. Each checkpoint evaluated the same fresh roots; results are not independent datasets.

The compact receiver builds the board from its own continuous local features and passes the original question to the frozen decoder. It never grafts state from the literal comparison. Same-scope irrelevant edits and other-scope edits each preserved all 72 matched packets, states and native logits per checkpoint. The guard refused 108 missing, wrong-scope or unauthorized requests and 435 deliberately mismatched source envelopes per checkpoint before native generation. Question-only produced zero correct pairs. The private wrong-value control produced the supplied wrong word in all 72 primary rows per checkpoint. These results do not cover learned identity lookup or production authentication.

**What it establishes:** in this bounded task, the board state left by the fact prefix was sufficient to recover the weaker decoder's missing answers. The controller is not shown to be irrelevant to *forming* that board or to later decoding. We zeroed or retained its state only after the prefix had already run. The packet stayed **224 bytes**, but the compact receiver still executes **14 continuous prefix positions** and native board computation, producing a transient full state of 10,208 bytes. The literal prefix is 14 bytes. This is not a latency, bandwidth, compression relative to text, or history-elimination result.

**Next:** test reuse of a verified, precomputed board state across repeated questions and corrections. Compare it with canonical text and live compact reconstruction on the same requests, counting time and bytes on both write and read paths before claiming a serving benefit.
