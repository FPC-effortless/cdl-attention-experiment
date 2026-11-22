# nanoCOG-001 — start of the attention-replacement implementation

Status: **CPU reference and protocol, not a trained language model**. Built in the CDL research repository as a reviewable branch rather than mutating the upstream baseline or presenting a fork that does not yet exist. [GitHub evidence index](RESULTS_INDEX.md) includes negative results and known custody gaps. [Qualification and stop rules](QUALIFICATION.md) are prospective.

## Implemented

```text
Tokens -> embedding + causal smear -> repeated blocks -> capped LM logits
                                     each block:
         current causal features -----+----> typed diagnostic updater (direct input)
                                      |
                           causal local mixer + bounded packet broker
                                      |
                                  residual + ReLU² MLP
```

- `LocalOnly` is the no-memory control. `PacketMixer` is an attention-free publish/subscribe reference with fixed per-layer buckets, learned topic/publish projections, hard forward routes and a straight-through gradient surrogate. It reads old packets and then publishes the current token; current features remain available directly. It does not claim semantic record keys. Training computes a dense routing surrogate even where reference inference selects a bounded topic. This is a Python/PyTorch reference, not a fused accelerator kernel.
- `PacketAndTypedProbe` additionally exposes a separate stable-address typed diagnostic path. Its supplied schema and exact executor are engineered. The path chooses hard discrete writes and currently needs separately trained routing heads or a validated estimator; LM loss alone does not train the discrete choices through argmax. The two paths do not yet jointly learn a common identity map. This prototype is not proof that the historical PULSE checkpoints learned stable binding.
- Functional decoder state supports full-sequence and chunk reference calls, reset with no state, and reuse of an unchanged prefix state for alternate continuations. Batches require aligned stream boundaries. These checks do not establish continuous batching, mixed-document isolation through the upstream packer, model-state provenance across reload, or native nanochat cache compatibility.
- Source-backed historical adapter: `nanocog/state/pulse_adapter.py` accepts an authorized already-instantiated PULSE-006 model. It reuses native current-event projections, publishes no predecessor implementation or weights and substitutes an explicitly engineered typed state/reader. Adapter test on exact historical source is optional; normal public tests use the local reference only.

## Upstream and fair comparison

Karpathy [nanochat at 92d63d4](https://github.com/karpathy/nanochat/tree/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd) is the baseline, with its license notice preserved. The reference decoder follows its common untied embedding/output, padded vocabulary, RMS normalization, embedding smear, per-layer residual scales, backout, ReLU² and logit soft cap. Rotary, attention-specific value embeddings, self-attention and KV state are not part of the candidate. Initialization, optimizer grouping, tokenizer loading, batching and full serving are **not** drop-in equivalents. Do not compare this reference directly to nanochat training or serving performance.

Next integration must wire the exact tokenizer assets, target and sample ordering, and compatible optimizer groups into a pinned nanochat checkout. The upstream `GPT.forward` consumes `kv_cache`; `Engine` creates and replicates `KVCache`. Add an explicit backend adapter rather than handing it a `DecodeState` and hoping its mutation/position semantics match. Carry reset, fork, previous embedding, row validity and session identity through prefill and each decoding step. Keep the upstream attention baseline intact. At a 32,768-token vocabulary, width 32 and untied embeddings, the embedding and head alone cost 2,097,152 parameters; use a feasible common budget instead of hiding embeddings to claim 1M.

## Checks and next decision

From this directory with Python 3.12, PyTorch 2.10 CPU and pytest installed:

```sh
OMP_NUM_THREADS=1 python -m pytest -q nanocog
```

These are engineering checks with generated inert tensors. No natural-text fit, paid compute, protected evaluation, end-to-end speed benchmark or architecture qualification ran. The [frozen-reader controlled-board diagnostic](PULSE_006_FROZEN_READ_RESULTS.md) since completed: hard reference-slot selection did not recover any convention-correct complete pairs. The next experiment should separate write fidelity from readout on exact stored payloads; a fresh constructed stable-key collision check remains a separate stage. Keep no-memory, unchanged PULSE and nanochat comparators; do not drop a failed seed or use the held qualification corpus to tune.

Workstreams: state and read invariants, sparse communication, independent experiment assurance, and integration/results custody. This is not a claim that all lab departments attended or endorsed the prototype.
