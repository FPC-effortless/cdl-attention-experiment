# COG Fusion 001

An executable research candidate joining continuous COG computation, learned
factual interpretation, protected current records, selective retrieval, actual
character generation and atomic continuation/recovery.

**Status: familiar controlled-history capability passed; complete learned-system
qualification failed.** Read [RESULT.md](RESULT.md) before using the checkpoint
as an agent. In particular, unfamiliar wording and observations after generated
answers remain unreliable. The full original language checkpoint also has
lower Shakespeare validation loss than the new candidate.

## Contents

| Component | File | Evidence boundary |
|---|---|---|
| Trainable recurrent COG and semantic/output interfaces | `fusion/model.py` | Six fixed primary fits |
| Supplied-source native implementation | `fusion/native/` | Exact checkpoint-bound hashes |
| Paired lifecycle generator | `fusion/data.py` | Controlled sixteen-identity vocabulary |
| Typed execution, bounded hybrid and generation | `fusion/engine.py` | Exact routes versus bounded similarity reported separately |
| Transactional records, provenance, packets and receipts | `fusion/store.py` | Conditional executor guarantees |
| Actual learned persistent session | `fusion/session.py` | Same interpreter and character head as primary candidate |
| Optional numerical computation | `fusion/composition.py` | Separately fitted, typed-address numerical task |
| Direct numerical-controller feedback into writer/read/output | `fusion/full_model.py` | Structural/gradient checks; not jointly language-trained |
| Matched primary fitting | `fusion/train.py`, `run_campaign.py` | Fixed budget, all endpoints retained |
| End-to-end assessment | `fusion/evaluate.py`, `run_evaluations.py` | Complete histories and generated answers |
| Ordinary language, live interaction and full serving cost | `fusion/preservation.py`, `fusion/interaction.py`, `fusion/benchmark.py` | Explicit baselines and limits |
| Independent audits | `review/`, `tests/` | Corpus replay, runtime crash/replay, causal diagnostics |

## Run a saved model

Use Python 3.12 and install `requirements.txt`. The recorded environment used
Torch 2.10.0+cpu, NumPy 2.3.5 and pytest 8.4.2; the scientific plots additionally
use Matplotlib. No GPU, external model service or historical prompt is required
for the CLI.

Extract the evidence package into this directory so the final checkpoint is at
`evidence/runs/memory_seed17/final.pt`. Seed 17 is a fixed representative, not a
post-test selection of the best result. The other two seeds are also included.

```bash
python -m fusion.cli --checkpoint evidence/runs/memory_seed17/final.pt --db demo.sqlite observe "Amara owns the chest."
python -m fusion.cli --checkpoint evidence/runs/memory_seed17/final.pt --db demo.sqlite observe "The coin is in the chest."
python -m fusion.cli --checkpoint evidence/runs/memory_seed17/final.pt --db demo.sqlite ask "Who owns the container holding the coin?"
python -m fusion.cli --checkpoint evidence/runs/memory_seed17/final.pt --db demo.sqlite status
```

Each command starts a new process and restores the matching committed state.
Use `--json` for interpreted operations, selected evidence and receipts. An
existing `--request-id` replays that historical response; use a new ID for a
question about current facts. Default IDs are new UUIDs. `--temperature 0.8`
uses the persisted private sampling generator.

The store accepts only the model's declared scope/model/schema continuation.
Changing a model requires a new database or an explicit migration; it does not
silently resume an incompatible state. The model's vocabulary is intentionally
small: Amara, Bela, Cora, Davi, Eren, Faye, Gita, Hadi, chest, crate, box, vault,
coin, gem, ring and key; relations owner and container. Generated spelling is
not restricted to those values, so incorrect text remains observable.

## Python integration

```python
from fusion.session import Session

with Session.from_checkpoint("evidence/runs/memory_seed17/final.pt", "agent.sqlite") as s:
    receipt = s.observe("Amara owns the chest.", "observation-1")
    result = s.ask("Who owns the chest?", "question-1")
    print(result["text"])

    s.store.mark_pending("edit-2", key=("world", "chest", "owner"))
    blocked = s.ask("Who owns the chest?", "question-2")
    s.observe("Bela owns the chest.", "observation-2", clear_pending=["edit-2"])
```

Pending-answer withholding is a runtime policy. Correct assignment of an
unfamiliar correction to a scope/key remains a learned semantic obligation.
The store cannot infer truth from the model's chosen destination.

## Reproduce qualification

```bash
COG_NATIVE_CHECKPOINT=evidence/reproduction_inputs/COG_CONNECTED_1M.pt python -m pytest tests -q
python review/review_data.py --corpus evidence/corpus.json --manifest evidence/DATA_MANIFEST.json --output fresh_data_review.json
python review/review_final_evidence.py
python -m fusion.evaluate --checkpoint evidence/runs/memory_seed17/final.pt --corpus evidence/corpus.json --output fresh_evaluation
python -m fusion.preservation --native evidence/reproduction_inputs/COG_CONNECTED_1M.pt --validation evidence/reproduction_inputs/val.bin --output fresh_language_preservation.json
python -m fusion.benchmark --output fresh_cost_results.json
```

Evaluation uses the fixed final checkpoints. DEV-only evaluation does not
silently invoke TEST interventions. Independent output files are required;
completed evaluation directories refuse accidental replacement.

The original training launcher contains the exact execution-environment input
paths in its receipt. For another machine, use the portable fitting entry point
with explicit inputs. The evidence archive includes the admitted native parent
and the pinned Shakespeare token files under `evidence/reproduction_inputs/`:

```bash
python -m fusion.train --native evidence/reproduction_inputs/COG_CONNECTED_1M.pt --corpus evidence/corpus.json --shakespeare evidence/reproduction_inputs/train.bin --output new_run --arm memory --seed 17 --steps 1000 --teacher-steps 250 --pairs 4 --cpu-cap 1800
```

Use `--arm no_memory` for the matched control. The archive retains final weights,
optimizer state, RNG state, data/source hashes, exact batch-order digests and
loss traces. Do not alter the original training lock or relabel an extended run
as the original endpoint.

## Optional read-compute-write attachment

`RecordComputer` reads two actual eight-dimensional numeric records, proposes a
derived value, and publishes it through `RecordRuntime`. The source identities
and destination are supplied. The separately fitted experiment teaches averaging;
it does not teach language to choose those addresses or request arithmetic.

`FullFusionModel` adds real controller-to-semantic-writer, next-read and original
character-output paths. Its zero controller preserves the v1 behavior exactly.
Only the 4,296-parameter numerical computer is fitted in the component study;
the additional feedback projections have structural tests, not joint language
qualification. See `COMPOSITION_PROTOCOL.md` and its recorded results.

Derived values must be accessed through `read_derived`, which validates both
support packets and interpretation identity. A raw database read of an envelope
does not qualify the value as current.

## Limits that matter

- Perfect familiar task scores do not establish reliability on arbitrary wording,
  identities, temporal semantics, long reasoning or multiple conversational turns.
- Restart equivalence preserves the saved process, including a wrong interpreted
  fact. It is not a semantic correctness guarantee.
- The strict path has no historical token attention or growing token KV cache.
  The hybrid still performs content similarity over records. Its reference index
  reconstruction is separately charged and is not constant total query work.
- Fixed neural-state size does not bound the durable archive or receipts.
- No matched Transformer, GRU or frontier attention training campaign was run
  here. The 12:1 inference-cost preference does not waive quality qualification.
- SHA digests and version guards target ordinary corruption and trusted-host
  mistakes, not a malicious process with unrestricted access to rewrite data/code.

