# NCM inference feasibility: bounded v2 outcome

17 November 2026. This is released-model/configuration software feasibility, not formal paper reproduction, CSM integration, trained repair, or architectural promotion. Author-source preprocessing review is complete: no source-level mismatch was identified in GPT2 tokenization, padding, generation settings, learned-VQ decoding or reranking. This is code inspection, not same-input numerical parity against the authors' runtime. Upstream dependencies are not pinned to this runtime; numerical/backend/version differences and the cause of the observed errors remain unresolved.

## Verified execution and retained-evidence audit

The revised safe loader completed a native ecosystem run on a free NVIDIA T4: seven generation calls (one cold, three warm repeats, three additional diagnostic calls), three learned-codec encode/decode reconstructions, and engineered copy controls. No training or protected TEST access occurred. Independent audit checked native receipts, artifact integrity and recomputed aggregate diagnostic scores from retained records. It did not independently repeat GPU inference or benchmarking.

| Observation | Result |
|---|---:|
| Exact learned-codec reconstruction | 1/3 fictional records |
| Literal generated field preservation | 0/9 scored fields across those three records |
| Warm complete-generation median, three repeats | 0.412541 s |
| Cold generation | 7.125171 s |
| Startup | 54.060281 s |
| Peak allocated GPU memory | 792,562,688 bytes |

Timing separates startup and cold work from warm complete calls. It is this small fixed released summarization configuration on this host, not a general latency bound. No optimized autoregressive or other matched comparator was measured; no inference superiority or paper speedup is established.

Codec reconstruction supplies the target text itself and tests codec fidelity, not historical recall. Literal-field omission in a summary is not automatically factual contradiction. Three fictional records cannot establish broad factuality, general language quality, or CSM memory competence. The evidence does not justify letting generated text overwrite authoritative state.

## Source and recovery

Executed v2 source is preserved at commit `b1bfcc2fe644b306416672a14db6815ff8d3aac2`. The reviewed loader and hash-bound launcher are in this directory; PUBLIC_SOURCE_SHA256.json identifies public file contents. Private raw prompts/outputs, native receipts, runtime evidence and archives are deliberately not published. Exact replay requires the owner-authorized private closure plus pinned external assets described in PROTOCOL.md; this public source alone is not a complete replay package.

The first attempt remains recorded as a pre-generation software failure: a blanket finite check rejected the authors' legitimate negative-infinity structural mask. V2 reconstructs and exactly validates that mask while retaining finiteness checks elsewhere. This failure was not a model-quality result and has not been erased.

WOYENGI-ENDTOEND-020 scientific training remains paused. This bounded NCM feasibility run is separate and does not count as a 020 fit. No new mathematical verification or CSM-conditioned training result is claimed.
