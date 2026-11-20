# Semantic transfer and write-admission qualification

This cycle tests the previous failure on unfamiliar role wording. The first experiment compares a frozen pretrained MiniLM encoder with a frozen random encoder of identical shape and the previous local role model. All arms train on identical bounded inventory transitions and support examples. The role encoders see the same masked operator spellings. A separate prediction head sees binarily labeled determinate and indeterminate commands. The transformer arms have matching role and clarity heads. The local clarity control is a learned token-sum model. Gold role and operator labels stay out of grounding training.

## Frozen encoder results

Each held-out row contains 1,536 commands across seeds 601, 619 and 643. Test phrasing is held out from the separately generated calibration data. Layout changes reorder shared role phrases. The implicit and correction cohorts introduce different clauses. The fixtures retain a supplied entity catalog, four supported operations and declared novel spellings.

| Exact action accuracy | Pretrained encoder | Random encoder | Local role model |
| --- | ---: | ---: | ---: |
| New layouts | 84.6% | 12.4% | 93.8% |
| Implicit roles | 43.8% | 6.6% | 33.3% |
| Corrections | 45.3% | 19.3% | 16.3% |

A separate admission head and threshold fit on calibration examples do not yield a useful write gate under new wording. The pretrained .80 confidence-only rule admitted 756/1,536 ambiguous examples. The calibrated rule admitted only 25/4,608 determinate commands, 24 correctly, and admitted 2/1,536 ambiguous commands. Coverage and exact precision fell short of the 70% and 99% gates. The fixed threshold and transfer results are recorded per seed in the audit. The pretrained model's input discipline and classifier are research prototypes. This experiment measures external pretraining transfer under matching encoder shape, and has no qualified architecture speed comparison.

## Entailment and clarification

The follow-up scores explicit hypotheses about who supplies the operand and where the result goes. Both arms use the frozen quantized DeBERTa NLI engine. The paired arm asks one joint hypothesis per valid binding. The separate arm combines source and destination scores. Both induce the verb meanings from 96 transition examples using their role scores. A separate NLI hypothesis predicts whether the binding is unresolved. Seeds and held commands are disjoint from the first experiment. The fixture uses 128 examples per seed and cohort. A prior compatibility smoke test checked one score vector without scoring its truth or adjusting the frozen model.

| Held-out outcome | Joint role hypothesis | Separate role hypotheses |
| --- | ---: | ---: |
| Implicit roles, exact | 147/384 (38.3%) | 289/384 (75.3%) |
| Corrections, exact | 178/384 (46.4%) | 241/384 (62.8%) |
| Calibrated determinate admissions, correct | 47/57 | 48/48 |
| Calibrated ambiguous admissions | 0/384 | 0/384 |

The separate scorer identified every intended operation. Argument selection caused the remaining exact-action errors. At the .80 confidence-only rule it admitted 3/384 ambiguous commands. Calibration rejected those and admitted 48 of 768 determinate commands with zero observed errors. Its 6.25% coverage missed the gate. The joint scorer admitted 10 incorrect actions after calibration. These are new cohorts, so the table supports within-experiment comparisons. A direct cross-suite comparison would require the same test commands.

In the single-slot clarification replay, choose the most uncertain of operator, source or destination, receive its exact value, and restrict the posterior accordingly. For the better separate scorer, one response per command increased exact accuracy from 530/768 (69.0%) to 752/768 (97.9%). It yielded 382/384 correct implicit actions and 370/384 corrections. The entropy rule asked about the source 200 times and the destination 568 times. Full confirmation resolves three slots and reproduces the 768 gold actions by construction. For the joint scorer, 325/768 rose to 704/768 after one response.

The post-response gate was explored after inspecting the joint clarification result. Its 99%-precision calibration cutoff accepted all 768 held-out determinate responses for the separate scorer, including 16 remaining wrong actions. It missed the 99% exact gate. The apparently resolved slot came from a truthful simulator, and the replay deliberately excludes ambiguous commands. These results quantify clarification value and residual risk. They leave real response accuracy and reliable autonomous commitment under novel phrasing open.

Research Lab OS registered 12 TrainLab jobs across the two protocols. The audits checked 20,736 held-out predictions and 8,448 calibration predictions. The Veritas ecosystem's separate reference comparison passed reverification.

## Reproduction

The pretrained checkpoints are pinned by revision and content hash. MiniLM uses revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41. DeBERTa uses revision a150876415327c80daeff35ca6f68f5ed8cf5c24 and the upstream AVX2 quantized ONNX variant. The bundle retains the checkpoints once each and references byte-identical snapshots with archive hard links. Upstream model cards accompany the weights.

Use Python 3.12, PyTorch 2.15 CPU, NumPy 2.3.5, transformers 4.51.3, safetensors 0.8.0, ONNX Runtime 1.24.4 and the Veritas ecosystem packages from eec439492407b0aa26922c23b5ae40d95289f4eb. From a fresh directory with the shipped recipe layout and scripts:

    (cd recipe/minilm && sha256sum -c ../../encoder.sha256)
    (cd recipe/nli && sha256sum -c ../../nli.sha256)
    export TOKENIZERS_PARALLELISM=false
    veritas-ecosystem run "$PWD/ecosystem"
    python launch.py
    python launch_nli.py
    python audit.py > audit.log
    python clarification.py > clarification.log
    python clarify_gate.py > clarify-gate.log
    veritas-ecosystem verify "$PWD/ecosystem"

The same-host audits check receipt integrity, recompute exact action and effect scores independently from saved outputs and fixture seeds, and rebuild the threshold choice from the calibration data. All sentence templates in the audit are constructed and share the same domain. These results leave open-ended grounding, independently learned competence parity and accelerator performance unresolved.
