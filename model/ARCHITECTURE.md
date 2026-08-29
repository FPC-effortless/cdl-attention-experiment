# CASM v0 — Compression-Attention Small Model

## Research question

Can a small causal language model trained from scratch learn a useful, efficient memory-routing rule when the routing score is trained to approximate **conditional description-length reduction** rather than relying only on Q/K geometry?

This branch treats compression as an **auxiliary structural signal**, not as a replacement for the main language-model objective. That choice follows the repository's Stage-B result: pure CDL imitation did not beat direct relevance supervision, so the new model keeps the task/language loss authoritative and asks compression to shape routing.

## v0 architecture

The intended full small configuration is ~5.2M parameters. The first CI pilot uses a ~1.5M-parameter configuration so architecture mistakes are cheap to falsify.

### 1. Raw-byte interface

Vocabulary: 256 byte values plus PAD/BOS/EOS/SEP. No pretrained tokenizer or pretrained embedding is used. This makes the experiment genuinely from scratch and avoids tokenizer-specific shortcuts.

### 2. Exact local computation

Each fixed-size chunk is processed with a modern decoder block:

- pre-RMSNorm;
- RoPE;
- grouped-query attention (GQA);
- SwiGLU feed-forward layers;
- tied input/output embeddings.

Chunking bounds dense attention cost; prior chunks are represented through compressed memory instead of an unbounded KV cache.

### 3. Low-rank compressed memory

After each chunk, the last valid causal hidden state is projected from `d_model` into a smaller `memory_dim` latent and written to a fixed ring. This is the explicit compression bottleneck.

The design is influenced by Multi-head Latent Attention's lesson that a lower-dimensional latent can reduce KV/state cost, but CASM uses the latent as a persistent cross-chunk memory rather than reproducing MLA exactly.

### 4. Persistent recurrent state

Two recurrent state slots are carried across chunks in addition to the finite ring. They are updated by learned retain/write gates. This gives the model a fixed-size path for information that must survive after detailed chunk memories have been evicted.

This is the minimal implementation of the user's persistent-state / verified-continuation research: transient token state, compressed episodic memory, and a distinct persistent recurrent state are not collapsed into one mechanism.

### 5. Dynamic compression attention

For every token in the current chunk, the model queries the previous compressed memories and persistent-state slots. The score is

`score(q,m) = normalized_q · normalized_k + c_phi(q,m)`

where `c_phi` is a small pairwise network. The resulting memory vector enters through a residual gate initialized near zero, so memory must earn influence rather than destabilizing the base model at initialization.

The ordinary-memory control disables `c_phi` and uses only the normalized Q/K dot product.

### 6. Compression supervision

At chunk boundaries, the same router is supervised by two sources:

1. **External cold-start compressor target.** Incremental gzip code length ranks retained raw chunks using the original compression-comparison idea: a memory is useful when conditioning on it makes the nearby continuation cheaper to encode. The external teacher decays but retains a small floor.
2. **Learned predictive code-length target.** A lightweight predictor estimates the NLL of upcoming bytes with and without each memory candidate. The difference is a candidate's learned compression gain. The target is detached before training the router.

The model's normal next-token loss still backpropagates through the memory path. Thus CASM is a hybrid objective, not pure distillation.

### 7. Multi-token prediction

Additional heads predict further future bytes. This is intended to increase sample efficiency and encourage representations that support short-horizon algorithmic continuation rather than only one-step token fitting.

### 8. Verified-continuation auxiliary

A small verifier distinguishes the true next chunk from a batch-shuffled continuation. It is an initial causal-consistency pressure, not yet a full verifier-gated write/repair mechanism.

## Why these mechanisms

The prototype intentionally does **not** combine every current architecture idea.

- GQA/RMSNorm/RoPE/SwiGLU provide a strong cheap decoder baseline.
- Low-rank memory follows the efficiency motivation of DeepSeek-V2 MLA.
- The local + compressed-memory split follows the coarse-compression/fine-selection direction seen in Native Sparse Attention.
- Fixed recurrent state is consistent with the state-compression direction formalized by Mamba-2/SSD and the explicit long-term-memory direction in Titans.
- Multi-token prediction is included because published work found sample-efficiency and algorithmic-reasoning gains.
- Dynamic byte patching (BLT), MoE, Mixture-of-Depths, Muon, and hard sparse top-k routing are deferred until the compression-attention mechanism itself survives ablation.

## Training tasks

The initial curriculum is deliberately heterogeneous rather than pure text memorization:

- associative recall with distractors;
- state tracking;
- arithmetic;
- rule induction;
- directed-graph reachability;
- reverse/copy transformations.

Hard/OOD variants increase sequence length, distractors, state-update count, and rule complexity.

The purpose is not to claim that a 1–5M parameter model can 'solve any problem.' No finite small model can guarantee that. The goal is to test whether the architecture learns a **task-agnostic computation substrate** that transfers across qualitatively different problem families.

## Primary ablations

All variants begin from the same compatible initialization.

1. `local-only`: exact local decoder, no cross-chunk memory use.
2. `qk-memory`: compressed + persistent memory with normalized Q/K routing only.
3. `compression`: same memory path plus compression-score network and hybrid compression supervision.

Primary metrics:

- LM NLL and byte accuracy;
- hard/OOD LM NLL and byte accuracy;
- answer-segment NLL / byte accuracy per task family;
- router entropy;
- compression predictor loss and estimated gain;
- wall-clock cost.

## Falsifiers

Compression attention is **not** considered successful if, after adequate training and repeated seeds:

- it does not beat Q/K memory on held-out/OOD task metrics;
- any gain disappears when parameter/compute differences are controlled;
- the compression correction stays near-uniform or unused;
- memory ablation does not causally change long-range performance;
- benefits occur only on surface-overlap retrieval but regress state tracking or compositional tasks.

## Later stages if v0 survives

1. Replace fixed chunks with entropy/surprise-based byte patches (BLT-like compute allocation).
2. Add compressed + selected exact anchors, closer to hierarchical sparse attention.
3. Train verifier-gated state writes and repair/replay controls.
4. Test recurrent/shared depth for parameter efficiency.
5. Test Muon after the architecture is stable.
6. Move from synthetic multi-task curricula to small public natural-language/code corpora and then to machine-experience episodes.
