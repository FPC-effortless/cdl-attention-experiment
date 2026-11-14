# FROZEN-VALUE-001 — errors precede lifecycle composition

14 November 2026. Exploratory frozen-checkpoint diagnostic; zero fits.

Some of the six LIFECYCLE-LEARNING-001 checkpoints fail a single SET→QUERY even when correct categorical controls and full write gains are supplied. Background writes and their order cause no answer changes on this grid. Overwriting the target does cause additional errors. This localizes a useful next question to the learned value transformation/readout path, but does not distinguish information loss in storage from a decoder failure.

## Results

Each cell gives correct answers out of 32 for seeds 101 / 202 / 303. Normal inference uses tokens alone. The aggregate is over 96 correlated key/value probes, not 96 independent experiments.

| Panel | CSM normal | Pointer slot normal | CSM aggregate | Pointer slot aggregate |
|---|---|---|---:|---:|
| Single SET→QUERY | 29 / 31 / 32 | 29 / 24 / 27 | 92/96 (95.83%) | 80/96 (83.33%) |
| Background first, target last | 29 / 31 / 32 | 29 / 24 / 27 | 92/96 (95.83%) | 80/96 (83.33%) |
| Target first, background last | 29 / 31 / 32 | 29 / 24 / 27 | 92/96 (95.83%) | 80/96 (83.33%) |
| Target overwritten | 26 / 32 / 30 | 29 / 23 / 21 | 88/96 (91.67%) | 73/96 (76.04%) |

Supplied-control results are identical except pointer-slot seed 303 overwrite: 20/32 instead of 21/32, giving 72/96 (75.00%). These privileged results are diagnostics, not learned success. Supplying controls also forces full gain; that combined intervention does not establish the isolated effect of any one head.

All 1,536 query executions preserved the four user records and their pointers exactly. All evaluated categorical decisions matched their masked process targets. This includes many noncommit zero-edit decisions and excludes unsupervised CSM operator choices. The six model state dictionaries had identical hashes before and after inference.

## What changed, and what did not

Across single, filled_last and filled_first, predictions are identical within every checkpoint and condition. Pointer-slot full logits match exactly, including normal inference. Under supplied controls, CSM logits also match exactly on those panels. Normal CSM logits differ slightly (maximum absolute difference 0.019586), without an argmax change. The preregistered pointer-slot context-invariance expectation is satisfied.

Relative to filled_last, normal overwrite harms 5 and rescues 1 CSM answers; it harms 8 and rescues 1 pointer-slot answers. These are paired counts over 96 probes per arm. CSM seed 202 improves to 32/32, so overwrite is not uniformly harmful across checkpoints.

Only 4/23 normal pointer-slot overwrite errors equal the immediately superseded value; all four are seed 303 targetv6 predictedv7. CSM has 0/8 old-value answers among its overwrite errors. The old value was deliberately absent from background rows. Thus most overwrite errors here cannot be described as literal stale-value responses. Lack of a stale answer does not establish absence of old-state interference in the internal computation.

Examples of errors already present in single SET→QUERY:

- CSM 101 predicts v4 for v6 on 3/4 keys.
- Pointer-slot 202 predicts v1 for v2 and v5 for v3 on all four keys each.
- Pointer-slot 303 predicts v7 for v6 on all four keys.

These class-specific confusions agree with several failures in the earlier lifecycle diagnostic, but do not prove that a class is globally unlearned.

## Interpretation and next experiment

Lifecycle composition, deletion and alias following are not necessary to produce these particular errors. Routing/control correction is insufficient to remove them. The current checkpoints have not demonstrated a reliable elementary value path, so a larger architecture-comparison claim remains premature. CSM's better observed accuracy in this limited comparison does not qualify architectural superiority or inference efficiency.

Both implementations compute a proposed SET representation using the current target's retrieved values. Even a full overwrite gain replaces the row with a proposal that can already depend on its old contents. This is a source-derived explanation for why a full gain is not an old-state-erasure guarantee; it is not yet a demonstrated cause of the observed errors.

The next bounded mechanistic test should freeze these same checkpoints and cross every old/new value pair, including same-value rewrites. Compare the normal learned proposal against zeroing only the target-read inputs to the SET proposal, with explicit control/gain conditions. Record target representations after SET and before QUERY, query-transformed representations, and logits. This tests the direct old-row inputs to the proposal, not every old-state pathway: CSM active-state controller/operator selection can still carry history, while a gain below one retains a separate previous-row interpolation term. Preregister that pointer-slot zero-read overwrite under supplied controls/full gain should match empty-row SET; CSM need not. Separate this proposal-input ablation from the gain intervention. It must remain an intervention diagnostic; no altered model should be credited as a learned success. A later equally supervised training repair can be justified by those findings, with an independent evaluation bank and fixed training budget.

## Scope

Four known keys, eight known payloads, new constructed history distributions, 128 fixed cases and six frozen checkpoints. Background values new+2,+3,+4 and old=new+1 modulo8 are distinct. The three filled panels have identical authoritative final maps. Single-versus-filled changes history, occupancy and context together; first-versus-last changes event order and retention context. No natural language, long-history scaling, general lifecycle success, throughput, energy or architecture qualification is established. The full probe grid is one correlated Forge subject; its TRAIN label is archival bookkeeping and no optimization occurs. No historical third-bucket examples are parsed or scored.

## Execution and provenance

Existing Forge→TrainLab→Veritas→Research Lab interfaces were used without infrastructure edits. Independent forward and backward semantic oracles agree on all 128 cases. Three source tests passed before inference, and the exact dispatch was independently approved. Independent post-execution review rescored all 1,536 saved predictions and replayed the complete 1,536-prediction assay, reproducing full logits exactly (maximum absolute difference 0), with unchanged weights and zero historical-data access attempts. This audit replay is not another independent research replicate. The successful TrainLab command made 1,536 predictions in 6.655586 seconds; this duration is a receipt, not a fair latency comparison. Runtime source/versions, parent source, six checkpoint hashes and weights before/after are preserved.

- Parent published source: a96ecc1b92fc7121bc35a8c60e1803328c05929f (draftPR58).
- Dispatch SHA256: 37f47499358d20caf9e6c9ef8a28e02a63ae08fcae596b8cdc4ac1539e3a505e.
- Freeze SHA256: fecd89f531cf03baca511cf0d1d0b02929d7843c1458edc99ef63d3b5f83cd8d.
- Forge manifest: 9ab36f429c6d7b858d6cad9507feff0b26223ada8ffffc7c7bc08903ad059bc1.
- TrainLab run: 20261114T052741-e7172ad7f5b0.
- Receipt SHA256: a705c0ed7e9d7f6e5a31b84c2f7a8518b62cf5d6ed12c3bb199341ac2f69d991.
- Predictions SHA256: a48d37f3c526cb5f03beca87b0cc0c89707c89c7b9450f4e92cb4d9bde33687d.

### Execution deviations

An initial preparation hashed historical parent dataset bytes through a broad verifier; it neither parsed nor scored them. That preparation was superseded before neural execution and retained. The executed source checks approved source/metadata/receipts/checkpoints only; an independent audit-hook guard confirmed no old dataset access in preflight.

The single inference command succeeded. Its reporting wrapper then rejected the 1.6 MB prediction JSON at the existing metadata reader's size limit. The frozen source and successful receipt remain unchanged. `finalize.py` verifies that receipt and reads the artifact with a bounded 8 MB standard JSON loader, recomputes the same aggregate, and imports evidence. Its first invocation stopped on a missing explicit `_runtime` import after source/run imports; the import was corrected and reporting completed. Neither reporting recovery ran a model or optimizer. No infrastructure limit was changed.
