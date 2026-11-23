# COG-1M-LATENT-002: compact handoff transfers two unseen values, but local state alone loses accuracy

**The 224-byte continuous handoff matched the exact local-state baseline on both checkpoints. One of two checkpoints passed the full gate.** The bridge had not seen the complete words `reed` or `fern` in training or development; it had seen each letter at its relevant position in other words. The frozen native text decoders already knew both words, so this tests transfer across the interface, not learning new vocabulary from scratch.

| Both contrasting answers correct | Checkpoint 20371 | Checkpoint 20471 |
| --- | ---: | ---: |
| Canonical text / exact carried full native state | 63/64 | 63/64 |
| Exact local state computed from text, with board and controller initially zero | 63/64 | 54/64 |
| Compact fitted handoff, same local-only initialization | 63/64 | 54/64 |
| Compact SVD initialization, no optimizer | 63/64 | 54/64 |
| Current question only | 0/64 | 0/64 |
| Compact fitted, values seen in bridge training | 61/64 | 59/64 |
| Compact SVD initialization, values seen in bridge training | 59/64 | 55/64 |
| Compact fitted, actual paired owner swaps | 63/64 | 54/64 |
| Compact fitted, new name spellings | 15/16 | 16/16 |

Keeping the full native state instead of dropping board and controller recovered nine net pairs on the weaker checkpoint. Fitting the bridge did not improve the primary paired score over fixed SVD. This points to the state that was removed, not a need for more optimization of this codec. The experiment does not distinguish loss of the board from loss of its controller. Word-order and unseen-symbol challenge panes yielded zero complete exact-answer pairs for both compact and canonical paths, so they do not establish arbitrary spelling transfer.

Same-scope irrelevant edits and other-scope edits each left all 128 matched packets, selected states and native logits unchanged per checkpoint. Actual serving rejected 192 missing, wrong-scope or unauthorized fixture rows and 1,524 privately tested wrong-source envelopes per checkpoint before native decoding. These are tests of the deterministic typed selection boundary, not learned identity or production authentication. The checkpoints use the same held roots and do not represent independent data samples.

The bridge used **7,470 trainable parameters** and actually passed seven ordered float16 factors (**224 bytes**) to its receiver, apart from local source metadata. It did not hand the receiver a rendered fact string. But the receiver still processed **14 continuous fact-equivalent positions** with the frozen local blocks and built a **5,856-byte local state**; the full native runtime state occupied 10,208 bytes. The 224-byte packet is smaller than the previous full-state handoff, not smaller than the 14-byte literal fact prefix. End-to-end actor runs lasted about 12 seconds each for all controls and 960 cases; those runs were not a per-method latency benchmark. There is no serving speed, storage, or prefill-reduction claim.

**Next:** carry or restore the smallest useful board/controller information and ablate those two components separately before retraining the codec. Test against literal text and full state using the same paired value and guard controls, then measure actual read-time savings.
