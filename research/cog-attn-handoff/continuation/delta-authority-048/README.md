# MultiWOZ delta verification, retrained 048 routing, and originating attestations

The three follow-ups use the established Veritas ecosystem and Research Lab OS. Twenty-four TrainLab jobs include 18 eligible module and signature jobs, three draft controls, and three draft attempts that stopped at a vocabulary error. The locally executed dialogue experiment and same-host audits are included separately. The Veritas public reference comparison passed reverification.

## 047, proposed state deltas instead of persistent state guesses

The candidate generator sees the latest user turn and immediately prior system text. The second model learns whether a particular proposed value is a new constraint. Its training positives are annotation changes and its negatives include candidate mentions and turns with no such change. The entailment arm scores whether the user asks for that hotel-area constraint. The gate looks at the runtime's own previous value to suppress no-op writes. Gold prior values were used to derive supervision and evaluation labels.

Both official test shards had been summarized in earlier work. The new methods and threshold grids were frozen before inspecting their predictions, with thresholds chosen on the two official development shards. The corpus is public MultiWOZ 2.2, pinned in sources.json. The test set has 3,213 user turns in hotel dialogues and 219 annotated hotel-area changes.

| Method | Admitted updates | Correct updates | Exact recall |
| --- | ---: | ---: | ---: |
| Raw score, without a confidence cutoff | 332 | 171 | 78.1% |
| Calibrated raw probability | 0 | 0 | 0% |
| Calibrated slot-delta model | 0 | 0 | 0% |
| Calibrated slot-delta plus entailment | 1 | 1 | 0.46% |
| Complete explicit directive plus slot-delta | 10 | 10 | 4.57% |

The raw-score diagnostic uses runtime no-op suppression, so its denominator differs from the previous single-shard headline. Entailment provided one correct admission where the two probability-only settings deferred everything. The post-hoc complete-directive contract selected a delta cutoff at .98, proposal floor .25, and no entailment minimum. The contract alone chose the same ten test writes. This is positive-coverage evidence on a narrow direct-language branch. It does not establish general-language commitment at 99% population precision. The 70% recall gate remains open. Earlier implicit role wording remains a separate bottleneck. In the previous held inventory cohort, even a perfect filter on 259/384 correct predictions has a maximum 68.0% coverage at 99% precision.

## 048, scoped replacement bridge

The original TAC IdentityFieldLayer code is byte-matched to TAC-Prime e51a921f82575d57ecf048e82a3898ca968a4224. The new fit learns its token embedding and cosine affinity parameters on typed family cues. A loss over that affinity path trains the branch and the actual forward method selects the bank at inference. The missing historical TAC checkpoint is not the fit used here.

The read stage runs the archived LRC repair checkpoints 901–903 on original supported input fixtures. LRC selects a destination via the two-hop query. The final stage uses a fresh original CSM recurrent reader plus a learned cue-to-mention scope scorer for the governing subject. The chosen mention becomes a canonical entity query to the original reader. The archived CSM commit predicate and typed original COPY evaluator check the selected subject, and SQLite commits the selected bank's value to the LRC destination.

| Configuration | Supported cohort exact | Challenge cohort exact | Writes without governing source |
| --- | ---: | ---: | ---: |
| Initial retrained CSM reader alone | 586/768 | 507/768 | 0 |
| Corrected scoped replacement | 768/768 | 768/768 | 0 |
| Original CSM control, matched corrected seeds | 374/768 | 350/768 | 0 |

The corrected arm deferred all 768 examples lacking a published family cue and all 2,304 unresolved-source controls. Held typed scope families share lexical cues and declared vocabulary with training. They test routing across new sampled worlds, not arbitrary language. Rotating each stage's selected result independently drove exact end-to-end writes to zero on the two supported cohorts. LRC independently recovered every archived two-hop target in the evaluated bridge banks. The checked checkpoints, exact sources, upstream commits, and code/weight hashes are in sources.json and module-manifest.json.

The first scope draft omitted a cue token and included a cohort-label shortcut for unsupported-family refusal. Its three candidate jobs stopped before evaluation. The corrected frozen adapter checks only observed input tokens. The audited result uses the corrected source digest. The original draft snapshot and failed receipts are preserved. This qualifies the declared replacement bridge on its typed fixtures. Historical TAC recovery and unrestricted-language module integration remain open.

## Originating keys and declared acquisition independence

The trust anchor issues an Ed25519 certificate binding an originating key to its acquisition family and era. Each signed reply binds that issuer, command, nonce, epoch, role, value and snapshot digest. Service aliases can forward signed evidence. They cannot mint a second family. Current matching evidence needs two registered independent families for each role. A local used-nonce set prevents replay within the verifier instance. The wire adapter bounds batches and rejects malformed input before verification.

| Adversarial condition | Signed service-name quorum, wrong accepted | Certificate-family quorum, wrong accepted | Digest-aware family quorum, wrong accepted |
| --- | ---: | ---: | ---: |
| Enrolled mirrors of one family | 192 | 0 | 0 |
| Same snapshot under two enrolled families | 192 | 192 | 0 |
| Different snapshots, concealed common bias | 192 | 192 | 192 |

Certificate-family verification rejected all unsigned aliases, modified conclusions, stale nonces, wrong origin keys and fabricated certificates across their individual 192-case cohorts. It admitted all 192 genuine independent cases. Requiring distinct snapshot digests also rejected 192 legitimate independent acquisitions with equal fixture digests. Signatures bind the issuer's claim. Correct domain enrollment and authentic acquisition remain separate facts.

In the silent-bias simulation, an independently supplied truth audit every eight commands detected 72/72 shifts after at most seven commands. This consumed 576 audit requests across 4,608 commands and reduced wrong pre-freeze decisions from 2,357 to 285. If the audit channel shares the bias it cannot provide that guarantee. Key enrollment, the acquisition intervention, and error schedules are fixture services. No network trust deployment or live evaluator was exercised.

## Replay

Use Python 3.12, NumPy 2.3.5, PyTorch 2.15 CPU, scikit-learn, cryptography with Ed25519, transformers 4.51.3, ONNX Runtime 1.24.4 and the Veritas ecosystem at eec439492407b0aa26922c23b5ae40d95289f4eb. The NLI study uses the same pinned quantized DeBERTa artifact as the earlier semantic-transfer study. Replay in a separate directory.

    (cd recipe/nli && sha256sum -c ../../nli.sha256)
    export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    veritas-ecosystem run "$PWD/ecosystem"
    python recipe/delta.py > delta.log
    python recipe/assertions.py
    python launch.py
    python launch_scope_verified.py
    python audit.py > qualified-audit.log
    python wire-check.py
    python replay-checkpoints.py > checkpoint-replay.log
    veritas-ecosystem verify "$PWD/ecosystem"

The archive includes the compact exact dialogue snapshots, original source shard hash list, signed fixture vectors, prediction receipts and trained model artifacts. The scope checkpoint's original sklearn vocabulary member uses NumPy object dtype. The replay reconstructs the deterministic feature columns and reads only numeric members, producing portable Unicode vocabularies under portable/. All comparisons are same-host. Independent scientific replication remains open.
