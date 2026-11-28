# Does grouping counterfactual questions teach COG to retrieve facts?

This experiment tests minibatch composition using the unchanged slow-state COG (arm C), grouped-query attention and GRU. Each model starts from its standard fresh initialization. Four fixed seeds (9731, 9743, 9769, 9791) each run in both conditions, for 24 fits.

## Intervention

A grid has two histories and two questions. The same two entities exchange their distinct digit values in the second history. The other two values are independently sampled and can coincide. We retain the symbolic and English generators used in COG-FACT-DIAGNOSIS-003. For each four-update block we draw eight grids sharing an unordered target digit pair. Each of the two values appears four times in every eight-example update. The two conditions have identical query frequencies, labels, domain, and prompt lengths per update. Every four updates they have seen exactly the same 32 prompt–answer examples.

- Grouped updates contain all four cells from each of two grids.
- Mixed updates contain one cell from each of eight grids. A rotation presents the remaining three cells in the next three updates.

The conditions only change which examples enter an optimizer update together. A model sees one sequence per row and receives no grid ID, other row, source annotation or gold intermediate state. Both use final-token cross-entropy with fixed real WikiText replay on the first update of every four. AdamW settings match COG-CONTEXT-BINDING-004. Each seed uses 256 updates with 2,048 factual prompts (256 unique grids per domain) and 32,768 next-byte replay targets. The matched comparison in this experiment is causal for this batching intervention. Cross-campaign comparisons also change the pool, objective and generator distribution.

Because final-answer cross-entropy is additive over samples, the two conditions have the same four-update average gradient if the parameters stay frozen. We verified this on one symbolic and one English block at initialization. The relative L2 differences are in gradient_check.json. Any benefit from grouping must arise through the sequence of parameter and optimizer-state updates, including clipping. This experiment does not test an explicit loss on relations among the four outputs.

## Evaluation and reproduction

The primary assay uses 192 separate held grids per domain, identical across all fits, with train and test divided by a digest of the canonical pair of histories. A strict success requires all four answers. The signed margin measures whether changing the query moves digit preference in the direction required by the changed history. The separate transfer panels score both revisions correctly on 64 paired worlds per seed for unseen names, phrasing and additional controls. BPB is measured on the same held corpus windows. Each results JSON includes the full panel data and separate checksums for factual training exposure and language replay.

In a checkout with CPU PyTorch and NumPy and the pinned TRACE corpus, run each arm, seed and condition from this directory. Create the results folder first.

```bash
python check.py
python train.py C 9731 grouped /path/to/trace_corpus results/C-9731-grouped.json
python summarize.py results
```

## Answer-frequency sensitivity

The primary experiment has two digit labels per minibatch. A secondary four-seed COG comparison uses blocks with four worlds from each of two complementary digit pairs. Each grouped update sees one world from each pair. The mixed rotation still presents one cell per world. All four digit labels occur exactly twice on each update in both conditions, verified at runtime. This comparison uses a fresh selection of worlds from the same generator, so the causal contrast remains grouped versus mixed within each pool. Use the additional `--four-digits` flag and a separate results folder to reproduce these eight fits.

## Findings

The 24 primary fits yielded zero complete symbolic grids and zero complete English grids for every arm and seed. Each primary arm averaged 23.7% symbolic and 25.4% English single-answer accuracy in both conditions. COG held WikiText loss was 4.519 BPB grouped and 4.574 mixed. COG did not change its answer on any of the 64 familiar English query swaps at any seed in either condition.

The eight additional COG fits also scored zero complete grids. In the four-digit comparison, mixed versus grouped updates yielded 30.3% versus 24.1% single-answer accuracy on symbols and 28.6% versus 24.2% on English. On the separate 64-world query-change panel the mixed runs changed their answers 122 times across seeds and got both answers correct 9 times. Grouped runs changed 31 times and got both correct twice. On novel-name revisions, mixed and grouped runs got both correct 10 and 1 times respectively. Both got zero paired revisions correct on novel wording. Mean held language loss was 4.079 BPB mixed and 4.226 grouped. These descriptive sums reuse the same held worlds across seeds. The prompts ask about A or B in the training grids, while the transfer panels sample the queried subject among four entities.

Grouping supplied no strict-grid advantage in either matched comparison. Mixed batches in the four-digit pool produced more sensitivity to the question but little correct switching. The four-digit sensitivity pool and the primary pool differ in selected histories as well as answer-frequency structure, so their absolute difference cannot identify a frequency-only effect. All runs start fresh, and each factual prompt appears once. Longer training, repeated exposure, and losses coupling the four answers remain open.

The independent audit reconstructs the training examples, checks labels against the existing parser, rules out overlap with held histories, and verifies the exposure digests. Run `python audit.py .` after producing both result folders.
