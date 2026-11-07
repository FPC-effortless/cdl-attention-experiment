# Learned relational composition — experimental prototype

**Status: implementation and CPU engineering pilot, not a qualified architecture.**
Read [the review](REVIEW.md) and [pilot protocol](EXPERIMENT.md) before interpreting
scores. Earlier LRR/CASM reports and code remain unchanged.

Each byte receives a bounded causal-convolution representation. Each layer
writes a distinct record per byte, predicts fixed-size discrete addresses and
relations, retrieves two ordered operands from a causal posting index, applies
one shared relation-conditioned MLP, and selects an occurrence reference to carry.
Only previous-layer values are read. Every batch row is a separate document;
padding must be a suffix. There is no prefix QK scoring or history aggregation.

| File | Responsibility |
| --- | --- |
| `index.py` | Append-only occurrences, strict predecessor lookup and candidate cap |
| `model.py` | Raw-byte LM, discrete choices, ordered composition and reference path |
| `train.py` | One bounded counterfactual per update, including writer choices |
| `data.py` | Templated text and evaluator-only semantic metadata |
| `run.py` | Paired CPU pilot, all-arm metrics, source hashes and resume checkpoints |
| `test_composition.py` | Mechanism, causality, cost and reproducibility tests |

From the repository root, in a Python 3.12 environment:

```bash
python -m pip install -r relational_composition/requirements.txt --index-url https://download.pytorch.org/whl/cpu
python -m unittest relational_composition.test_composition -v
python -m relational_composition.run --output /tmp/lrc-pilot --steps 20 --seeds 101 102 103
```

The dependency file pins the CPU-tested PyTorch version. The prototype does not
need the Gradio/Transformers application dependencies. Output paths must be fresh,
including when resuming. Checkpoints include model, optimizer, RNG, step, training trace,
configuration and source hashes. A resume must use the same source/configuration,
seed and batch size; `--steps` is the total target step, not extra steps:

```bash
python -m relational_composition.run --output /tmp/lrc-resume --steps 40 --seeds 101 --variants coupled --resume /tmp/lrc-pilot/coupled-101.pt
```

For arbitrary text, use `data.encode(text)` to construct each document row and
train `model(tokens)` against the one-byte-shifted targets. `forward()` receives
no semantic metadata. Evaluation is deterministic argmax routing; training uses
categorical Gumbel draws. The local counterfactual loss is a biased heuristic,
not a demonstrated solution to long-delay credit assignment.

All three variants have the same parameter shapes. `independent` separates read
and transform relation choices. `no_reference` removes the continuation feature
path. `memory="disabled"` and `memory="shuffled"` are inference controls. The
shuffled control rotates retrieved identities within the earlier prefix, retaining
the same gather count. At a one-record prefix this necessarily has no effect.

The original LRC-001 path prioritizes inspectable correctness. It has Python CPU indexes,
host transfers, no incremental decoding cache, no natural corpus result, and no
attention/SSM performance baseline. Appending records prevents destruction but
does not guarantee the relevant occurrence falls inside the read cap.

## LRC-002 information diagnostics

The follow-up [protocol](EXPERIMENT_LRC002.md) adds trained local controls,
explicitly privileged oracle endpoint access, answer-only diagnostic losses and
an independent random-answer control. It reports answer bits and free-running
two-byte generation separately from whole-text loss. Run the frozen matrix with:

```bash
python -m relational_composition.run_information --output /tmp/lrc002-evidence
python -m unittest discover -s relational_composition -t . -p 'test_*.py' -v
```

`oracle_operands` is exclusively a capacity diagnostic: it already resolves the
supporting relation chain and must never be reported as learned routing. The
original LRC-001 source and result are preserved on `research/relational-composition-v01`.

## V3/V4 inference integration

The separate inference paths add cached evaluation of LRC-001 (`StreamingInference`),
dependent two-hop reads (`SerialMultiHopLM`), and historical page selection
(`PagedMultiHopLM`). See the [integration review](../results/inference-v4-review/REVIEW.md)
for provenance, fixes, fresh measurements, limitations, and publication status.
V3 remains the faster reference path; V4 remains experimental. Neither is
qualified by learned multi-hop accuracy. The 12:1 inference/training cost
preference applies only after competence and expressivity requirements pass.

```bash
python -m pip install -r relational_composition/requirements-inference.txt
python -m unittest discover -s tests/inference -p 'test_*.py' -v
python -m experiments.inference_v4.benchmark_paired --repeats 8 --output /tmp/lrc-paired.json
python -m experiments.inference_v4.stress_v4 --output /tmp/lrc-stress.json
```

The original supplied reports remain in `results/inference-v4-import/`; fresh
integration measurements are in `results/inference-v4-review/`. Their runtime
versions differ. Lean sources are under `formal/relational/`; kernel verification
is pending. From that directory, a working pinned Lean installation should run
`lake build Relational.Basic Relational.Paged Relational.Programs`.
