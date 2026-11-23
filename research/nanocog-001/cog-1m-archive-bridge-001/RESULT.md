# COG-1M-ARCHIVE-BRIDGE-001: persistence works, factual answering does not qualify

We tested the already-trained **993,574-parameter Tiny Shakespeare COG** with the original persistent runtime and a separate accepted-fact archive. No training, weight changes, or answer rewriting were performed.

**Cold continuity passed; learned factual use did not.** With the same restored state and current question, independently committing a relevant archive edit did not change any initial logits or greedy output in **64/64** comparisons. The original model path does not read the archive. Warm and fresh-process native output and model inputs matched on all 64 cases, with identical restored recurrent/board/controller tensors, logits and RNG and no model forwards during restoration. Traced model inputs had **zero old prompt tokens** in both cold paths. The explicit bridge adds a marked current record as *new* input. The zero claim concerns prior prompt replay and logical old source access, not all memory reads or physical SQLite page I/O.

| Input path | Correct base + edited pairs | Correct individual answers |
|---|---:|---:|
| Native restored model, archive disconnected | 0/64 | 0/128 |
| Restored model + explicit current accepted record | 0/64 | 0/128 |
| Immediate record and query, empty recurrent state | 0/64 | 0/128 |
| Complete raw history replay | 0/64 | 0/128 |

Each pair required the model's unconstrained greedy generation to choose the exact uppercase A/B answer under displayed choices, correct on both sides. It is not free-form fact extraction. The panel had 16 synthetic fact families, crossed with both edit directions and two question wordings; the 64 comparisons are not independent samples. Both wording strata failed identically on the registered correctness endpoint.

The engineered bridge looked up one current record by its externally supplied exact key. It changed **64/64** initial logit arrays and **6/64** eight-character generated strings when the relevant value changed. Both unrelated edits and accepted same-value rewrites preserved logits and decoded strings **64/64** each. Disabling the bridge matched native behavior **64/64**. Even examining only A-vs-B ranks as a secondary diagnostic yielded **0/64** correct pairs in every path. The expected current facts were not supplied as precomputed answer letters. Separately, disabling the native memory board or feedback each changed logits in all 16 checked model inputs, but changed the first greedy character in none. Those are causal data-path and numerical signals, not correct decisions.

Observed median per-query application time on one CPU: native restored **35.6 ms**, restored with bridge **47.5 ms**, and complete replay **64.5 ms**. The bridge looked up the accepted record in **3.62 ms** median and injected 14,432 new characters across the 256 all-world invocations. These figures exclude one-time model/process startup, are descriptive from one host, and are **not a speed advantage at competent task accuracy**. There is no measured KV traffic or broad-query retrieval guarantee here.

**Implication.** The long-lived runtime can continue without replaying previous prompts on this fixture, but correct use of changing information remains unestablished for the trained model. Since the immediate-fact control also failed, these negatives do not isolate a long-delay retention bottleneck. Qualify immediate and short-history task competence first, then test delayed changes, restarts, and total cost against capable alternatives.

Exact checkpoint SHA-256: d8bab3d1d99a43dfb6cca2257f2caae4d3a7282d62b079c92874ce736a44ec3c. Archive implementation: [exact MemoryStore source](https://github.com/FPC-effortless/cdl-attention-experiment/blob/ffb842adc176496d8838b774d0fe412aec5d1d20/research/nanocog-001/nanocog/memory/store.py). The separately delivered original test kit supplies weights; no weights are reproduced in the replay evidence. Independent raw recount verified all 37 registered aggregate fields (receipt SHA-256 521eda9ab1a80111feffbaaab2fafd5154bcc640c3625ea438adf306316b7526).
