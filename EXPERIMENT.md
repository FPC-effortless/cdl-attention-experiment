# Experiment specification

## Hypothesis

A memory's relevance to a query can be approximated by conditional description length:

`R(M,Q) = -L_model(Q | M)`

and a cheap Q/K router may later be trained to approximate this richer target.

## Stage A: no training

For every benchmark case:

1. generate one relevant memory and hard distractors;
2. score each memory with gzip;
3. score each memory by SmolLM2 `-NLL(query | memory)`;
4. optionally measure native last-layer attention mass;
5. select top-1 memory for each method;
6. calculate SmolLM2 answer NLL conditioned on the selected memory.

## Falsifiers

Treat these as negative evidence:

- CDL does not outperform gzip;
- CDL retrieval is below native attention by a practically meaningful margin;
- CDL retrieves the labelled memory but does not improve answer NLL;
- CDL advantage disappears as candidate count grows;
- wins depend on near-verbatim templates only.

## Stage B (only if Stage A passes)

Generate teacher scores `T_i = -NLL(Q | M_i)` and train a small student router:

`S_i = q_phi(Q)^T k_phi(M_i)`

with distribution matching:

`KL(softmax(T/tau_t) || softmax(S/tau_s))`.

Compare against the same student architecture trained directly on relevance labels.

## Stage C

Inject the distilled score into a small language model's routing/attention path and test perplexity, long-context retrieval, compute, and robustness on natural data.
