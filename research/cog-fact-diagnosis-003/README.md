# COG fact-binding diagnosis

The three 1,077,494-parameter slow-state COG checkpoints preserve substantial evidence of a recently supplied fact. They fail most reliably when the question must select which preserved value controls the answer. English also exposes an earlier architectural limit in the protected-memory write controller. This study measures facts supplied in context. Learning permanent world knowledge in weights, factuality in open-ended generation and multi-hop reasoning require different experiments.

## Following a fact from input to answer

We used the existing COG-BALANCE arm C and its balanced trace-plus-pair repair checkpoints, seeds 9731, 9743 and 9769. We froze all weights. Each history contains four authoritative key–digit records. In a second history, keys A and B exchange digits. Both questions are posed to both histories, giving four answers per grid. The two histories contain exactly the same characters with the same frequencies. Guessing the most common digit cannot solve the grid. In English, four shuffled official statements compete with four independently valued rumors. In the independent-distractor assay, the A and B values are distinct by construction. C and D can independently repeat any digit. The symbolic weak claim is about D and occurs last in this assay.

Each offline linear reader fits on 384 grids and evaluates on 192 different grids. Counterfactual siblings always stay together. The train/test split is disjoint by canonical history within the diagnostic, including swapped histories. Each checkpoint sees the same evaluation grids. A keyed reader has a different set of weights for each queried key. A shared reader must produce the requested digit with the same weights for both questions. Neither supplies anything to COG during inference. Percentages below are single-answer accuracies unless specified.

| Independent-distractor symbolic assay | Seed 9731 | Seed 9743 | Seed 9769 |
|---|---:|---:|---:|
| Requested digit from history, keyed reader | 91.3% | 99.7% | 100.0% |
| Same reader with shuffled training labels | 24.1% | 27.7% | 23.0% |
| Query identity from final hidden state | 100% | 100% | 100% |
| Digit from first block after query, keyed reader | 45.7% | 49.2% | 49.5% |
| Digit from final hidden state, keyed reader | 50.5% | 54.4% | 51.3% |
| Digit from final hidden state, shared reader | 41.9% | 42.6% | 41.5% |
| COG's own next-token answer | 24.7% | 28.1% | 34.6% |
| Answer changed when requested key changed | 0.0% | 28.1% | 13.8% |
| All four native answers correct | 0/192 | 0/192 | 0/192 |

With the same independently valued distractors in English, the keyed history reader reached 52.5%, 63.9% and 75.7%. The final hidden state still identified the question perfectly. COG answered 24.0%, 27.6% and 30.5% correctly, with zero complete grids for each seed. In symbolic and English trials, the supervised intermediate question positions also yielded zero complete grids. The symbolic key position itself yielded 24.7%, 31.9% and 35.2% correct. The answer had already become unreliable by the time COG saw the query key.

An untrained COG allowed 100% keyed symbolic history decoding and 36.3% English decoding on the same independent-distractor tests. High symbolic probe scores show that an external reader can extract a signal from the state. They cannot establish that training gave the model a working retrieval algorithm. Recoverability from the larger collection of persistent states and from a single output-facing vector are different measurements. The scores locate a functional bottleneck without quantifying irreversible information loss.

## The protected memory path

At each position, the native controller determines a role, slot and payload from a five-byte window. In the familiar English template, the digit in “The log lists Ada at 2” has the local input ` at 2`. Ben, Cia and Dan give exactly the same five bytes. The rumor template gives that window too. Across all four digits and three checkpoints, changing the name caused exactly zero change to the controller's role, slot or payload logits at that digit. The file template showed the same name blindness at its digit. In symbolic notation, changing the key in `!a2` did change the controller logits. The full recurrent model has other paths that can carry older information.

On the symbolic permutation assay, writes and resets touched only two of the eight protected slots in each checkpoint. Seed 9769 issued zero payload writes across 5,760 history positions, even though it used reset and explicit-read roles. Swapping only the protected state between counterfactual histories changed the original-versus-alternative answer logit margin by exactly zero in all three checkpoints. The keyed protected-state reader scored 25.1%, 26.3% and 22.1%. A model can also consult the background memory under the background role, or expose a protected slot on a write. Explicit-read counts alone cannot characterize memory use.

| Mean absolute change in answer logit margin after a state swap | Seed 9731 | Seed 9743 | Seed 9769 |
|---|---:|---:|---:|
| Entire history state | 0.00097 | 0.08752 | 0.40221 |
| Fast recurrent state only | 0.00047 | 0.08297 | 0.06313 |
| Continuous background memory only | 0.00034 | 0.00578 | 0.32934 |
| Protected slots only | 0 | 0 | 0 |

These interventions hold the question and model weights fixed. Full-state swaps use valid states. A component swap can produce a hybrid state and effect sizes need not add. Magnitude measures sensitivity and does not guarantee a correct direction of change. The active path differs by seed. There is no single downstream matrix established as the cause.

## Retention, load and depth

In a separate assay, we changed one initial digit and followed it past three independently valued distractors. A keyed reader recovered that digit from the first block's fast state on 100% of examples immediately after presentation, then 81.6%, 87.6% and 82.4% after the third distractor. At the answer delimiter these values were 69.9%, 82.9% and 78.4%. This fixed-key test detects declining accessibility with delay, while showing a residual trace. Each history in this assay has a fresh five-letter prefix.

| Direct test, both versions correct out of 384 | Seed 9731 | Seed 9743 | Seed 9769 |
|---|---:|---:|---:|
| One fact, immediate question | 0 | 0 | 206 |
| Three intervening records, one distractor identity | 0 | 2 | 47 |
| Three intervening records, three different identities | 0 | 2 | 49 |
| Three other identities, target record last | 0 | 9 | 100 |

The matched-length comparison changes the number of distinct distractor keys. Its small differences give little support to physical slot saturation at this load. Two checkpoints already failed with one fact. The third benefited from recency and lost most of its paired successes after intervening writes. Those writes also increase delay relative to the immediate condition, so the experiment cannot fully separate decay from overwrite. The single-record and repeated-key panels differ from the five-claim training distribution.

In the archived matched-data depth comparison, the 2-layer and 22-layer arm-C parents have 1,077,494 and 9,999,932 parameters, the same embedding width of 128, and matched training digests over 384 steps. At seed 9769 the smaller parent answered both versions of 10/128 official retrieval pairs, the deeper parent 0/128. Language loss was 2.9842 and 3.4825 bits per byte, respectively. These are parent checkpoints before balanced repair. A fixed-step depth comparison leaves width, number of independent slots, optimization to convergence and larger training budgets unresolved.

## Why next-byte language loss can improve

We scored the same 8,192 next bytes from held WikiText windows for each condition. The train-only byte-frequency predictor scored 4.600 bits per byte. The repaired COG checkpoints scored 3.736, 3.631 and 3.646 with the original prefix. Replacing the older 85 of 89 prefix bytes with unrelated held text, while keeping the last four intact, gave 3.734, 3.633 and 3.651. Replacing all 89 gave 4.079, 3.991 and 4.012. Each window scores eight consecutive teacher-forced targets, so later targets also see up to seven additional true bytes. Paired differences and window standard errors are in the language results.

For these checkpoints and targets, much of the language advantage is available from recent context. Selecting the right value for a named entity is a different demand. Corpus-average loss does not measure the quality of that selection. The original 1M parent showed a larger penalty of 0.035 BPB on the four-byte-prefix condition, and its absolute language loss was lower than the repaired checkpoints'.

## Training and architecture controls

The same diagnostic also reproduced the gap in two original Native HH parent checkpoints, before the balanced repair stage. For seeds 9731 and 9743 the keyed history reader reached 99.7% on symbols and 71.6% and 70.7% on English. Neither parent ever changed its native answer when the question changed on either panel. Both scored zero complete grids. This makes an explanation confined to the slow-state side path or the later repair objective unlikely.

All twelve COG checkpoints spanning final-only, trace, paired revision, and trace-plus-pair objectives across the three seeds scored 0/192 strict symbolic grids with independent distractors. Matched grouped-query attention and GRU controls also scored 0/192 at all three seeds. Their single-answer ranges were 24.9–38.0% and 23.4–34.2%. Hence a training or task-distribution limitation shared across architectures remains plausible. The observed five-byte protected-write restriction is specific to this COG path. Archived ordinary single-fact-revision pairs gave 0/64, 2/64 and 11/64 paired successes for the repaired COG checkpoints, so a strict-grid zero does not imply every individual revision failed.

Across 32 fresh paired minibatches per primary checkpoint, the mean cosine between the real-text loss gradient and the original combined factual training loss gradient over all weights was −0.0006, −0.0050 and −0.0021. Thirteen, seventeen and sixteen minibatches respectively had negative cosine. These endpoint measurements give little evidence for strong, systematic opposition at the checkpoints. They cannot reconstruct earlier training dynamics or rule out competition in individual features. The balanced repair used 256 updates and 2,048 question streams per seed, with real-text replay on every fourth update.

## Scope and reproduction

The strongest inference is that the learned process for selecting a stored value in response to a question is unreliable. English provides direct evidence of an additional write-site limitation. Longer contexts can weaken stored traces. These results establish neither a universal COG capacity ceiling nor the origin of errors on general factual QA. The 192 grids repeat across models and should not be pooled as independent samples. The original campaign BPB, the parent BPB and these targeted measurements use different checkpoints or scoring windows.

All scripts run on CPU with Python, NumPy and PyTorch. The sibling research directories provide the exact native model, corpus reader, training loss and parsers. The suite uses original checkpoint files and the pinned TRACE-007 corpus. Checkpoint hashes and parent readouts are in `checkpoints.json`. The matched tests and raw per-seed outputs are in `out/`.

```bash
python run_suite.py --repair-dir /path/to/extracted/repair --scale-dir /path/to/scale-archives --corpus /path/to/trace_corpus --out /tmp/cog-diagnosis
python verify.py /tmp/cog-diagnosis
```

The verifier reconciles the native head scores from two independently implemented scoring paths, checks direct versus streamed output parity, and checks held grid labels against the existing task parser in all four domain/sampling conditions. The scripts fit diagnostic readers, swap internal state for measurement, and compute gradients. They make no update to model weights or inference code.
