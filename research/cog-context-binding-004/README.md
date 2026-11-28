# COG protected-memory write and read experiment

This experiment crosses two changes to the 1,077,494-parameter slow-state COG. The writer adds the existing fast recurrent representation to role, write-address and payload logits before the hard write. The reader adds a gated, query-dependent mixture of the eight protected slots to the native message. The two changes reuse existing recurrent channels and projection rows. Neither requires new parameters, extra persistent state, a parser, entity IDs, or a whole-history attention cache. The writer changes a shared role decision that also gates the background bank. A writer effect therefore includes that routing change.

The unchanged native COG, writer, reader, combined arm, grouped-query attention and GRU start from their standard fresh initializations at seeds 9731, 9743, 9769 and 9791. Within each seed, the four COG arms start with identical tensors. This tests acquisition from scratch at a fixed budget. It does not measure repair of the previously trained checkpoints. Each fit uses 256 AdamW updates with the existing balanced trace-plus-pair objective on eight question streams per step. Every fourth step also uses four real-text replay windows. The factual inputs and trace targets and the WikiText replay windows are hashed. The model receives ordinary prompt bytes. Trace and revision labels enter only the training loss.

Each evaluation grid poses two queries before and after swapping the corresponding facts. The original and revised histories have identical byte counts for every symbol. Primary metrics are the number of fully correct grids among 192 and the mean signed logit interaction. The interaction is the difference in digit preference when the query changes, differenced again when the facts are swapped, divided by four. Positive values indicate correct binding. Names, phrasing, rumor edits and query changes have separate paired 64-world confirmation panels. Each fit also scores 16 held batches of real text. Latency is indicative because jobs share the host; validate a serving claim under isolated CPU measurement.

From this directory run each combination as follows, substituting the arm and seed. Dependencies are NumPy, CPU PyTorch, the sibling research directories in this checkout, and the pinned TRACE-007 corpus. The results directory must already exist.

```bash
python check.py
python experiment.py both 9731 /path/to/trace_corpus results/both-9731.json
python summarize.py results
```

The implementation checks compare the inactive variant to native COG, then compare split-prefix and full-sequence logits with each intervention in training and evaluation modes. The result summarizer checks data digest agreement, native COG parameter parity, streaming output parity and all 24 expected results.

## Results

Each seed saw the same 192 held grids in each domain. The strict counts below sum successes across four fits. The four runs reuse the held worlds. Single-answer scores and language loss are four-seed means.

| Arm | Symbol grids | English grids | Symbol single | English single | WikiText BPB | Logical state bytes |
|---|---:|---:|---:|---:|---:|---:|
| C | 0/768 | 0/768 | 31.1% | 26.5% | 4.277 | 3168 |
| write | 0/768 | 0/768 | 31.7% | 28.5% | 4.233 | 3168 |
| read | 0/768 | 0/768 | 29.3% | 28.1% | 4.299 | 3168 |
| both | 0/768 | 0/768 | 32.9% | 27.7% | 4.282 | 3168 |
| GQA | 0/768 | 1/768 | 37.2% | 30.1% | 4.646 | 49184 |
| GRU | 0/768 | 0/768 | 36.9% | 25.3% | 3.942 | 1056 |

All COG variants scored zero strict grids at each seed in both domains. The attention control solved one English grid at seed 9743. The writer improved average English single-answer accuracy from 26.5% to 28.5% and paired accuracy on new phrasing from 7 to 14 out of 256 model–world evaluations. Its signed symbolic binding margin was nonpositive at every seed. The combined arm had mixed margin signs. The improvements in isolated metrics do not qualify a replacement backbone. All data-digest and stream-parity checks passed.

## One-grid positive control

A separate run with seed 9731 repeatedly presented one symbolic history pair crossed with both questions, using ordinary final-answer cross-entropy. Each of the six arms reached four correct answers for five consecutive updates. Native COG took 81 updates and solved 8/64 held grids afterward. The combined arm took 227 and solved 3/64. Only two distinct answer digits occur on the training grid. This measures capacity to fit a particular reversal. The primary experiment and positive control also differ in repetition, language replay, loss and input distribution. A matched comparison must manipulate the crossing of histories and questions separately to identify its causal effect.

Run the control with `python overfit.py C sanity/C.json` after creating the output directory. All six result files are in `sanity/`.
