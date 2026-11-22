# nanoCOG stable typed records: causal contract

This is an engineering diagnostic, not a trained replacement for the PULSE reader. Default schema follows the PULSE-005/006 taught coordinate convention: four box records, four cargo records; owner values 0–7 are terminal, and values 8–11 are stable links to boxes 0–3. The schema is a declared structural prior. There is no claim that the model learned the ontology, supports an unseen/open record set, or discovers record identity from unlabeled text.

## Packet and transition

For each batch row maintain immutable stable addresses \(r\in[0,R)\) and state \(S_t=(v_t,n_t,a_t,c_t)\): typed values with empty sentinel \(-1\), revision counts, last material change step, and live-event clock. An independently generated packet is \((r,o,u)\), with op HOLD, SET or CLEAR. A SET is legal iff address and value are in range and the fixed compatibility relation admits the type. CLEAR requires a valid address. HOLD ignores address/payload. Padding accepts no packet and advances no clock. A present invalid packet changes no record but consumes one visible step. A legal write of an identical value (or clearing an already empty location) produces no revision.

Define \(M_i\) iff the row is live, the packet is legal and material, and \(i=r\). Then
\[
(v'_{i},n'_{i},a'_{i})=
\begin{cases}
(u^*,n_i+1,c),&M_i \cr
(v_i,n_i,a_i),&\neg M_i,
\end{cases}
\qquad c'=c+\mathbf{1}_{\mathrm{live}},
\]
where \(u^*=-1\) for CLEAR and \(u^*=u\) for SET.

The implementation creates new tensors, never mutating caller state. “Atomic” means all fields of **one packet** apply together to its one addressed record or none apply. It is not a multi-writer database transaction, a persistence barrier, or an authenticated source. Validate untrusted imported/restored state with `board.validate_state`; a Python frozen dataclass is not a read-only tensor capability.

## Read, locality and causality

Read a nonempty typed row by stable address. A terminal value resolves immediately. A link uses its fixed destination address, reads that row, and requires a present terminal value. No additional link is followed; empty/invalid/unresolved returns an explicit invalid result, not owner zero. The optional second read therefore identifies the target independently of changing payload meaning. Changing a foreign row cannot change a fixed first-address read unless that row is the requested record or the link destination used by that read.

For the generic sequence wrapper, all projections are per-token. The output at step \(t\) is read from the state **before** applying its packet. Inductively, if two valid input sequences have identical prefixes through step \(t\), their input state, current proposal, address and output through \(t\) are identical (for the same fixed weights/runtime and execution order). The new state follows deterministically from those same quantities and cannot encode a later step. The current input may influence where it reads, not what it already wrote. PULSE integration’s last sentence is the query only and does not advance or mutate persistent state. This is strict temporal causality, not a claim that the chosen slot answers the query correctly.

Integer state/chunk/fork parity is exact for identical hard decisions. Floating projection implementations may choose a different hard decision when logits are at a tie/roundoff boundary across device, batch shape, dtypes or kernels; no general bit-parity claim across those changes. Non-padded hidden inputs and learned weights are required to be finite. Invalid padded hidden positions are zero-sanitized before projections and never update state.

## Integration

- `TypedRecordBoard.apply(state, WritePacket(...), active)` → `(new_state, ApplyResult)`; `read(state,address,active)` → `StateRead`.
- `StableRecordMemory(width)(x,state,valid=mask,return_state=True)` → `(delta,state)`. Default returns `delta`; optional `return_aux=True` appends the raw learned routing/write logits and decision receipts. The caller owns norm/residual composition. State has 25 int64 elements per default sample = 200 bytes in this interpreter, excluding allocator/static schema/module state. This engineering loop visits all fixed slots per discrete update and is not optimized deployment performance.
- `Pulse006TypedAdapter(pulse)` takes the actual archived PULSE class as an injected module and accepts the native token tensor/lengths. It preserves the native encoder, observation channel write/subscriber/broker, current-event merge, commit/address/value heads and first query lookup; it then decodes fixed PULSE taught coordinates and resolves the typed direct/link record. Native learned second read and final classifier are **intentionally not** reused. The typed candidate and native answer logits are different endpoints, not a zero-effect intervention or comparable pretrained checkpoint result. No archived source, data or model is included with this component. Optional test imports exact source only when explicitly supplied.

## Learning and scope limits

The hard argmax packet and hard read address intentionally stop answer-loss gradients to key, routing, operation and write heads; only read payload embedding and its output transform learn from this path. Auxiliary raw logits enable explicit comparator-matched supervision. A differentiable surrogate or mixed routed gradient is a separate experiment, not a default replacement for the exact guarantee. A successful exact interpreter proves neither raw-text access nor model capacity; genuine attribution requires the ordinary actor (no annotated addresses/operations), matched frozen native and typed controls, separate structural and answer metrics, supervision disclosures, intact/no-effect and state-removal interventions on a new admitted cohort. Count learning and inference cost separately.

## Engineering checks

Run from the directory containing the `nanocog` package: `python -m pytest -q nanocog/state/test_records.py`. An optional source-only test requires `PULSE006_SOURCE` pointing to the archived original source tree. It validates the pinned source hashes and synthetic PULSE token inputs with forced test-head parameters, not any trained model or held dataset.

Checks cover typed address/value rejection, repeated idempotent SET and CLEAR, unresolved/dangling and multi-link rejection, invalid state validation, 512 generated packets against a pure interpreter, future-prefix invariance, step/chunk and duplicate-row forks, padded NaN immunity, explicit stopped-gradient behavior, and optional native source integration. No external services, paid compute, protected data, or scientific model training is required.
