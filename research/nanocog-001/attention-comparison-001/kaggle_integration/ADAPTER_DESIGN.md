# Pinned model integration: source-only engineering stage

This module is a proposed integration for independent review. **No tests, native model execution, optimizer updates, learned quality comparisons, or hardware speed measurements were performed by its author.** Execution receipts must be appended separately by the runner. The source does not admit the scientific campaign.

## Sources and ownership

The candidate is the original `nanocog.model.Decoder` with `PacketMixer` or `LocalOnly` from candidate PR88 at `39be434ff160d4e6507df8d2114979ba5529caff`. LocalOnly is a separately parameterized control intended for independent training; no arm here has been trained. The attention baseline is the unchanged `nanochat.gpt.GPT` from `92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`, configured with actual GQA and full context (`window_pattern="L"`). The comparison proposal source is PR90 at `1a91168b216a49543cc839153fa299a262320730`.

The launcher must verify the pinned source checkouts. `build_model_triplet` additionally records SHA-256 of the actual imported model/cache source files. These observed hashes supplement the declared commit identities; they are not proof that a mismatched checkout is acceptable. No upstream source or canonical candidate source is changed by this integration.

The adapter author owns `adapters.py`, `test_adapters.py`, and this design document only. The independent reviewer must review their exact completed bytes before execution. Review of an earlier protocol is not review of this implementation.

## Interface

```python
models, provenance = build_model_triplet(
    vocab_size=32768, width=128, layers=2,
    n_head=2, n_kv_head=1, max_seq_len=256, seed=1729, device="cpu",
)
# keys: packet, local, gqa; all start in eval mode, all are untrained
model = models["packet"]
loss = model(tokens, targets, loss_reduction="mean")
session = model.new_session(batch_size=tokens.size(0), max_tokens=256, session_id="request-1")
logits = model.prefill(tokens, session, session_id="request-1")
next_logits = model.step(one_token_per_row, session, session_id="request-1")
fork = model.fork(session, row_indices, session_id="request-1", new_session_id="fork-1")
report = model.state_report(session, session_id="request-1")
model.reset(session, session_id="request-1")
```

The stateless loss interface always resets candidate state. It calls the original candidate with `targets=` by name and unwraps its result; otherwise a positional target would be misinterpreted as decoder state. Both arms accept ignore index `-1`. Targets are made contiguous symmetrically because unchanged GPT uses `targets.view(-1)` while the candidate uses `reshape`; ordinary next-token slices with multiple rows otherwise fail only in the baseline. Any necessary copy belongs to the adapter's measured cost. Common stateless calls require more than one input token because of upstream training-forward semantics; single-token cold starts use `prefill`.

Serving accepts aligned, unpadded rows, one complete prompt, then exactly one token per row per call. All rows have the same position and lifetime. An explicit validity mask containing any invalid token is rejected; without tokenizer assets, the adapter cannot infer whether a numeric token ID was intended as padding. Mixed positions, padded batches, continuous batching, partial row reset, speculative decoding, tool calls and serialized continuation are not supported.

## Common initialization and differences

`GPT.init_weights()` runs unchanged. Candidate/local embedding, output projection, every MLP matrix, residual scales, x0 scales, smear gate/gain and backout gain are then copied exactly, into separate parameter storage. Upstream smear/backout scalars have shape `[1]`, while the original candidate uses `[]`; these two explicit one-element mappings reshape without changing values or source parameter geometry. The source-name mapping is emitted in provenance and checked for bitwise value equality and independent storage. A common seed alone is insufficient because construction order and initializers differ.

Attention-specific rotary, Q/K/V/output projections, value embeddings and value gates remain in the baseline. Candidate/local mixers retain their native initialization and are separately seeded. GQA dimensions and packet defaults are visible in configuration/source. The candidate does not acquire a token-attention or SSM path. Equal width/depth does not imply equal total or active parameter count; the parameter report explicitly declares that capacity matching has not been established.

The parameter-role inventory covers every unique registered parameter, with counts and shapes. Common parameter roles reference the upstream optimizer groups; attention-specific value embeddings and matrices remain distinct. Packet biases, age weights, packet matrices and local depthwise convolutions are identified as requiring candidate-specific optimizer admission. **No optimizer is instantiated.** In particular, upstream's assumption that its block parameters are suitable Muon matrices is not silently applied to candidate bias or convolution tensors.

## Owned decoding and state identity

Prefill uses the original model forwards. Candidate state is copied once at the ownership boundary. Each packet decode call uses the original `PacketCommunication.step_inplace_` in eval plus inference mode. A small delegated view invokes the original `Decoder.forward` while supplying an owned mixer call; it does not duplicate the decoder equations or patch source methods. LocalOnly uses its native functional local-state update. Previous embedding remains part of the exact decoder state.

Every session is bound to a particular adapter instance, caller-specified session identifier, state schema, parameter/buffer signature, declared model config and attention windows. Ordinary in-place parameter/buffer changes, optimizer updates, model reload/replacement and device moves invalidate existing sessions. Construct models outside inference mode so version counters exist. Direct `.data` mutation, changes to other private model implementation attributes, direct writes to private state internals, concurrent operations on one session and external aliasing of state tensors are unsupported. A failed owned decode poisons that session because an earlier layer may already have mutated its cache; create a new session after such failure.

Forks use independent storage, including duplicate selected rows and previous embedding. The parent is retained unchanged. Whole-session reset releases retained cache references. Baseline cache allocation includes its entire declared capacity; allocation is not confused with the number of populated positions.

The unchanged baseline's cached `T>1` continuation omits the preceding embedding at the first token of later chunks when smear is nonzero. This adapter rejects every multi-token cached continuation. Full prefill followed by single-token decode is the supported baseline boundary. Candidate native full/chunk equivalence is tested separately. A future upstream correction must have a separate implementation identity.

## State and cost accounting

`state_report` enumerates every retained state tensor, its logical bytes and underlying storage bytes. Unique storage addresses are deduplicated, so retained views count their full allocation while aliases do not inflate totals. It includes KV capacity, cache positions, packet local/payload/presence/position arrays and previous embedding. A separate recursive `sys.getsizeof` estimate includes Python tensor wrappers and metadata such as tokens seen, session identity and model signature. It is labeled an estimate, not exact Python allocator accounting.

Model parameter and buffer storage are reported separately. Returned logits, temporary allocations, prefill/fork copies, allocator reserve/fragmentation and peak process/device memory are outside retained state and must be measured by the runner. Reset releasing references does not imply a CUDA caching allocator returns reserved memory. These reports do not establish end-to-end latency, throughput, memory superiority or a production serving implementation.

## Deterministic engineering checks

`run_checks(device)` returns a JSON-serializable PASS/FAIL report with failures retained. It checks exact common-weight copies, common loss/ignore-index semantics including noncontiguous next-token target slices, suffix causality, vocabulary-logit agreement for full versus prefill/single-step execution, persistent packet/KV storage preservation across owned steps, native candidate chunk agreement, unsupported baseline chunk rejection, fork independence, reset, session/model mismatch, stale weights/buffers, padding rejection, refusal after an injected partial owned-update failure and retained storage formulas/capacity changes. An activated GQA fixture must respond to changes in distant history, so passing parity with a dormant attention branch is insufficient.

Native initialization has deliberately zero projections and zero smear gain. Its gradient audit records these zeros without treating them as broken connectivity. A **separately labeled intervention fixture**, with nonzero smear/output projections and deterministic packet occupancy, requires finite nonzero gradients on every claimed trainable parameter. The gradient fixture uses three layers so the backout tap precedes an actual remaining block; a two-layer shell's final-tap scaling is almost cancelled by final RMSNorm and would make a backout-gradient check depend on numerical epsilon. It modifies weights for a software test, creates no optimizer, performs no optimizer update and produces no trained checkpoint. The same nontrivial fixture prevents zero initial projections from making cache and causality tests vacuous.

## Dependencies and admission gaps

Set `NANOCHAT_DTYPE=float32` before Python imports. This first adapter explicitly rejects other upstream compute dtypes. Put the exact nanochat checkout and `research/nanocog-001` on `PYTHONPATH`. The actual module imports Torch and upstream engine/GPT dependencies. Upstream package metadata pins Torch 2.9.1; a different runtime must be recorded and validated as a distinct engineering environment. Upstream imports can discover/download external kernels; the launcher owns internet policy, dependency pins and actual backend receipts. An API named `flash_attn` does not prove a FlashAttention kernel ran.

No tokenizer assets or real-text datasets are invented here. A synthetic integer vocabulary tests interface semantics only. Common real tokenizer bytes, `token_bytes.pt`, authorized data/license manifests, exact packing/order/resume, mixed precision, optimizer groups/schedules, gradient accumulation, checkpoint lifecycle and complete production trainer integration remain pending. The upstream best-fit packer reuses buffers and resumes approximately; paired training must clone/materialize its consumed schedule or implement exact resumption under review. Its BOS packing is not document-isolated attention/state.

The unchanged upstream `scripts/base_train.py` is a top-level GPT-specific CLI. This adapter reuses model/cache source and exposes the common loss contract; it is not a rewritten or fully integrated trainer. The existing statistical comparator is also separate from training. Scientific training, held evaluation, quality gates, competent control budgets, tuning fairness and full costs require their own exact-version admission under `QUALIFICATION.md` and the comparison protocol.
