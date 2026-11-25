# COG-1M-LATENT-001: a learned state handoff matched the text prompt on known values

**The decoder did not need an English fact prompt for this task.** A separately trained 265,328-parameter bridge converted seven characters from an already verified typed record into the native recurrent state of an unchanged 993,574-parameter COG decoder. The decoder received the original 11-character question and generated a complete word plus period with its original, unrestricted 65-character head. It did not receive the fact as text at inference. Both frozen decoder checkpoints passed the preregistered gate.

| Comparison | Checkpoint 20371 | Checkpoint 20471 |
| --- | ---: | ---: |
| Learned state, held-out name/value pairings | 64/64 | 64/64 |
| Canonical fact text / exact state carried from that text | 64/64 | 64/64 |
| Same raw question, no fact | 0/64 | 0/64 |
| Learned local recurrent state only | 64/64 | 61/64 |
| Learned board and controller state only | 0/64 | 0/64 |
| Learned state, new name spellings | 16/16 | 16/16 |
| Learned state, two values withheld from the bridge | 0/16 | 0/16 |
| Canonical text / exact carried state, same two withheld values | 16/16 | 16/16 |

A pair counts only if both contrasting answers are exactly right. Each checkpoint used the same roots; they are not independent evaluation sets. Both bridges got 256/256 TRAIN rows exact on unrestricted free decode. Exact state carried from canonical text preserved every output, tie and stop condition, with up to 7.40×10⁻⁶ absolute logit drift from chunk boundaries; it is not free because forming it still processes the fact prefix. Swapping which owner received the two known values preserved 64/64 complete causal pairs per checkpoint. Injecting the wrong value for the same owner in a private diagnostic made it generate that wrong value on 128/128 primary rows per checkpoint; the real serving gate rejected all 1,204 tested mismatched source envelopes before decoding. Changing another same-scope record or the opposite scope preserved the selected state and native logits on 128/128 rows each per checkpoint. All 192 missing, wrong-scope or unauthorized fixture requests per checkpoint were refused without native generation.

**What this establishes:** The verified-record-to-decoder handoff can be continuous instead of English, and nearly all the signal survived in local recurrent state without the board and controller. It does not show that the model learned identity lookup, writes or long-term memory; typed identity/scope/permission checks selected the record before the bridge. The bridge saw the typed value, learned from canonical prefix states and answer supervision during TRAIN, and did not transfer to two unseen values even though the frozen decoder could generate them from text. The full state was 2,552 float32 values (10,208 bytes), compared with 14 ASCII characters of fact prefix. This is neither a storage compression nor a serving speed result.

The two original native FIT workers completed optimization and emitted the exact checkpoints; the native records remain marked failed because the post-run observation file omitted its required metadata envelope. A separate independent, model-free verification confirmed all original numeric workload conditions, and approved scoring two successful blind evaluations of those exact checkpoints without retraining or rewriting failure records.

Next test: learn and evaluate a substantially smaller local-state transfer, with whole value words held out from the bridge, before claiming an open-vocabulary or efficient memory interface.
