---
title: Conditional Description-Length Attention Test
emoji: 🧪
colorFrom: indigo
colorTo: blue
sdk: gradio
python_version: "3.12"
app_file: app.py
pinned: false
---

# Conditional Description-Length Attention Test

A falsification-first experiment for the hypothesis that relevance can be defined by conditional description length rather than only learned Q/K compatibility.

## Model

`HuggingFaceTB/SmolLM2-135M`

## Primary score

For memory `M` and query `Q`:

`CDL(M,Q) = - mean_NLL(Q | M)`

Higher is better. This treats a memory as relevant when conditioning on it makes the query less surprising / shorter to encode under the language model.

## Baselines

1. gzip incremental compressed bytes
2. native last-layer SmolLM2 attention mass from query tokens into the memory prefix
3. oracle relevant memory
4. query-only downstream answer NLL

## Anti-cheating design

The benchmark uses counterfactual entity/value names and generates a fresh randomized world. Each case has exactly one supported relation and hard distractors:

- same entity, wrong relation;
- same relation, wrong entity;
- correct answer value in the wrong fact;
- random plausible facts.

It intentionally avoids contradictory same-entity/same-relation facts, because a target-free router has no principled way to know which equally asserted contradiction the evaluator secretly marks correct.

## Main metrics

- Top-1 relevant-memory retrieval
- Top-3 retrieval
- MRR
- downstream answer NLL after selecting a memory

A routing method only counts as useful if its selected memory improves prediction, not merely if it matches a relevance label.

## ZeroGPU deployment

Create a **Gradio** Space and select **ZeroGPU** hardware. Current Hugging Face ZeroGPU applications require GPU-dependent functions to be decorated with `@spaces.GPU`; this app does that around the benchmark call.

With the current `hf` CLI, an eligible free account can use a command of this form:

```bash
hf repos create ngrnblud/cdl-attention-test \
  --type space \
  --space-sdk gradio \
  --flavor zero-a10g \
  --public \
  --exist-ok
```

Then copy this directory into the Space repository and push it.

## Recommended first run

- 120 cases
- 6 candidate memories
- native attention enabled

Then rerun with 12 and 24 memories. Do not tune prompts/templates after seeing results without reporting that tuning.

## Decision rule

The hypothesis earns a Stage-B distillation experiment only if CDL:

1. beats gzip clearly;
2. is competitive with or better than native attention retrieval;
3. selects memories that reduce answer NLL;
4. does not collapse when candidate count increases.
