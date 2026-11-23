# COG-UNIFIED Tiny: implemented language-model path

This is the **text-only trainable instantiation** of COG-UNIFIED-001 for the
user-requested Tiny Shakespeare checkpoint. Its architecture identifier is
`cog-unified-tiny-latent-v1`. It is a native causal model with a learned latent
record board, not a Transformer and not a complete learned semantic executor.

The preceding proposal and abstract proofs are pinned at
`FPC-effortless/cdl-attention-experiment@5c42980a555b3253949e5f556a9af725bd3c34c3`,
under `research/nanocog-001/unified-001/`. This new model is an implementation
candidate. The Lean statements do not certify these PyTorch operations or the
learned behavior of the checkpoint.

## Concrete default geometry

| Component | Default |
|---|---|
| Tokenization | Character IDs supplied by the training-only vocabulary |
| Width | 400 |
| Residual blocks | 6 |
| Feed-forward expansion | 2 times width, with SwiGLU |
| Latent key factors | 4 entity-coordinate classes by 4 relation-coordinate classes |
| Latent board | 16 slots, each with a 64-dimensional value |
| Forward query/write route | Exactly one factorized categorical address |
| Forward operation | VALUE replacement or HOLD |
| Recurrence/read accumulation | Float32 |
| Training dropout | 0.1 by default; disabled for generation/evaluation |
| Parameters for vocabulary size 65 | 9,725,682 |

Each added vocabulary character contributes 800 parameters because the input
embedding and output head are separate. There is no padding with unused
parameters to achieve the target size.

## Causal local blocks

Each block first normalizes its input and adds a learned channelwise amount of
the previous normalized input. The latter is a one-token causal feature, with
an explicit carried state at chunk boundaries. A learned projection supplies
three vectors: a retention coefficient, a value innovation, and an output gate.

For each coordinate,

\[
a_t=0.90+0.0998\,\sigma(d_t),\qquad
b_t=(1-a_t)\tanh(v_t),\qquad
h_t=a_t h_{t-1}+b_t.
\]

The coefficient and innovation depend on the block input and its previous input,
not on the same block's evolving hidden state. They therefore admit an affine
prefix scan. The block emits a gated projection of the hidden state through a
residual connection, followed by a normalized SwiGLU residual network.

The current implementation uses parallel doubling with O(T log T) work. It is a
portable PyTorch reference, not a work-optimal custom accelerator kernel. It
accumulates the scan in float32. Associative reordering introduces ordinary
floating-point differences; full-sequence, chunked, and token-by-token outputs
must be checked with numerical tolerances. Discrete routes near a tie can be
more sensitive than the local numerical difference.

The retention range is a numerical design choice. It does not give exact
unlimited memory or prove a trained model will use long-range information.

## Latent record interface

The final causal features feed independent query-entity, query-relation,
write-entity, and write-relation heads. Their hard categorical choices define
one query slot and one proposed write slot. A commit head chooses VALUE or HOLD.
A value head constructs a bounded vector that the read-value head later converts
back to the shared output feature space.

For hard write indicator w[t,s] and vector value v[t], the forward board update is

\[
M_{t,s}=(1-w_{t,s})M_{t-1,s}+w_{t,s}v_t.
\]

Each w is exactly zero or one in the forward computation. Only one slot may
change at a step, and all slots remain unchanged on HOLD. This update also has
an affine scan because the proposals are derived from causal local features.
The query at t reads M[t-1], **before the current write**, and contributes to the
same learned language output head as the local branch.

The board's entity/relation coordinate names express factorization of an address
space. They do not establish that an address means a human entity or relation.
The stored vectors are learned latent values, not verified factual records.
Unannotated Tiny Shakespeare does not provide gold support, semantic operation,
binding, or preservation labels. This run trains these interfaces through the
next-character loss and the explicitly stated surrogate below; it does not
silently invent semantic supervision.

## Gradient estimator

Categorical addresses use a hard forward one-hot and softmax straight-through
derivative: `hard + (soft - soft.detach())`. The binary commit uses the analogous
sigmoid straight-through derivative. This is a biased estimator, not the true
derivative of argmax or thresholding.

The softmax is over a fixed categorical address vocabulary. There is no
token-to-token query/key matrix, no attention over token history, and no borrowed
SSM implementation. Dense categorical and board surrogate work during training
is still real work and belongs in training cost. Even hard forward execution is
currently evaluated with dense small-board tensor operations, rather than an
optimized sparse gather/scatter kernel.

Nonzero residual projections and read gates permit the language loss to reach
the read/write interfaces from initialization. Gradient checks must establish
the observed path in the actual implementation. A connected gradient is not a
guarantee of useful learned memory, correct credit assignment, or convergence.

## Mapping to the unified proposal

| Unified component | Status in this checkpoint path |
|---|---|
| Raw tokens and common output interface | Implemented with character embeddings and a learned output head |
| Causal local recurrence | Implemented with six affine-scan blocks |
| Bounded hard routing | Implemented for a factorized 16-slot latent board |
| Direct current-event features | Implemented: write heads receive the current causal backbone features |
| Aligned value writer and reader | Implemented and trained jointly through the language objective |
| Stable full semantic keys | Not learned or claimed; addresses are stable latent coordinates |
| Semantic support/write/binding supervision | Not available in this raw-text run |
| Exact typed SET/COPY/LINK/DELETE/transactions | The preceding semantic executor remains a separate executable interface; not in the learned token path |
| Bounded semantic LINK traversal | Not implemented in the latent board and not claimed |
| Versioned evidence archive and provenance | Not implemented in this token path |
| Separate publish/subscribe packet broker | Not implemented as an additional subsystem; the latent board provides the bounded communication experiment |
| Pure typed resolution cache | Available in the preceding semantic reference; not a neural-output cache here |
| Demand-aware procedure compilation/search ordering | Optional future domain adapters; not trained by this corpus |
| Selective weight editing with preservation checks | Not part of this fit |

Thus this run tests whether a roughly ten-million-parameter causal backbone
with jointly trained latent record writing and reading can learn a small text
corpus and support usable text generation. It cannot establish the whole
semantic fusion, general language understanding, long-term factual reliability,
or superiority over attention. Those claims require different data, controls,
and integration work.

## State and API

`ModelConfig` is a serializable frozen dataclass. `COGTinyLM(config)` exposes:

```python
logits, state = model(input_ids, state=None)
state = model.init_state(batch_size, device=None)
state = detach_state(state)  # only at a declared gradient-truncation boundary
```

Inputs are nonempty `torch.long` tensors of shape `[batch, sequence]`; logits
have shape `[batch, sequence, vocabulary]`. Passing no state resets the entire
stream. A state contains the hidden and prior-input vector for each block and
the latent board. Callers own batch order, document resets, and model identity;
states must not be reused across unrelated prompts or models. Generation uses
the same forward function with one token at a time in evaluation mode.

Default persistent tensor state is 23,296 bytes per stream in float32
(6 x 2 x 400 local scalars plus 16 x 64 board scalars). This excludes model
weights, logits, temporary activations, allocator overhead, and saved states.
It is not a bound on total process memory. Sequence training retains intermediate
scan activations and has a larger memory requirement.

`memory_enabled=False` provides an inference intervention that leaves the local
path unchanged and skips the learned board contribution. It is useful for
measuring checkpoint dependence on that board, but it is not a separately
trained local-only baseline or an architecture comparison.

## Compatibility boundary

The new language path does not mutate previous experiment source or checkpoints.
The prior exact typed reference remains separately reproducible. No claim of
representation compatibility with historic COG/CSM checkpoints is made. The
Tiny Shakespeare fit is an exploratory generation artifact, not admission of a
new confirmatory architecture-comparison campaign.
