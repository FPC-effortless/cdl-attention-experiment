# COG-ATTN-011–013: semantic grounding, feedback efficiency and guided planning

Exploratory follow-up to 002–010. No general-purpose LLM qualification. Three seeds (11,29,47) for language studies, 32 procedural planning seeds (100–131). Training/test templates and pair split are defined in source. No prospective external preregistration or independent replication. All tasks use controlled labels or simulated environments. No GitHub publication.

## 011: role grounding

15 fits. Train 160 optimizer updates, paired training examples in every arm, 64 examples per batch (two views). Sparse CNN 6,364 parameters; one-layer message-level attention encoder 5,980 parameters. Both use top-8 of 24 latent activations, learned positions and role supervision. One-template arm has a reduced training template set, but all other conditions share six templates. Alignment is MSE on same-role vectors across supplied paraphrase views. Entity pairs with `(a+b)%5==0` withheld from training. Each evaluation uses 1,024 examples per seed, per cohort. No whole-history task or raw-token generative modeling measured here.

| Arm | Familiar | All six training templates | Held entity pair across six | Held syntax | Both | Unknown words |
|---|---:|---:|---:|---:|---:|---:|
| Single CNN |100|16.57|16.11|0|0|49.54|
| Diverse CNN |100|100|100|36.78|36.88|4.13|
| Aligned CNN |100|100|100|55.96|56.38|18.95|
| Attention |100|100|100|16.31|16.34|5.76|
| Aligned attention |99.97|99.97|100|8.50|9.99|0.26|

Exact whole-message role tagging; not answer accuracy. Unknown words are mapped to one UNK token. Single-template control's 49.54% unknown-word result is consistent with one of the two novel templates sharing the training role positions; not vocabulary acquisition. Held paraphrases include a previously unseen non-role word only if mapped to UNK; these are primarily novel role order/context. Neural baselines are trained on equal steps but not tuned to convergence or compute-matched, and differ in encoder architecture; cannot infer attention's inferiority in general. Same test cohorts evaluated across arms; no independent external benchmark. 90 checkpoint/cohort aggregates replayed locally.

## 012: consequence-efficient action discovery

BANKING77 official PolyAI split, 10,003 train, 3,080 test, 77 labels. Source https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data ; CC BY 4.0 (Casanueva et al.). Git blobs: train 98e2543cf482d0dca7bfb175ebe35d98efad95be; test 799687a8367359432985b8b13d85a2baf73f92dd. Reward oracle reports whether proposed random-permuted action ID matches dataset intent; no real action. All policies share shuffled training order, action permutation and word/bigram IDF derived on unlabeled train utterances. Positive feedback updates prototype counts; no update on failure. Ranked policy uses sparse cosine scores; explicit exploration prefers least-observed actions after first ranked action. Random policy guesses without replacement. Each utterance presented once. No held-out paraphrase identification in banking data. Under cap 77 both discover all annotations and learn identical count prototypes; speedup is feedback calls, not time. Three-attempt novelty control was exploratory after initial capped collapse, not preregistered. Ranking may scan all 77 actions; no claim of scalable bounded retrieval. Dataset not redistributed. Results executed using fetched CSV contents in connector JS runtime; reward.mjs reproduces same algorithm; syntax checked locally, not executed against CSV locally.

| Policy | Cap | Mean feedback calls | Top-1 | Mean discovered actions |
|---|---:|---:|---:|---:|
| Random |77|390293.33|82.50|77|
| Ranked |77|26650|82.50|77|
| Random |3|29619.67|48.77|76.33|
| Ranked |3|28879.67|5.94|4.67|
| Ranked |1|10003|2.60|2|
| Mixed |3|21766.67|59.84|72.67|
| Novelty |3|20541.33|66.81|77|
| Warmup |3|24433.33|65.66|73.33|

## 013: sparse-cue-guided planning

32 random 256-state toroidal worlds with permuted action IDs, 20% transitions independently masked. The planners see only observed transition IDs; no coordinates or hidden transitions. Eight landmarks, directed forward/reverse distance caches, four selected by goal distance. A* uses admissible directed triangle-inequality lower bound on distances when finite. This is engineered graph association, not learned neural synaptic branch scoring. 2,035 reachable pairs out of 2,048 generated, 13 excluded because even uncapped search on the observed graph failed. Every returned path verified against true simulator and optimal BFS length. Both arms share transition-check budget; scoring overhead differs. Full query timer includes heuristic vector construction across all 256 states (not a truly bounded sparse search); preparation separately charged. Mean prep 14,726.25 transition references, 4.228 ms per graph, plus 2*8*256 int32 distance entries (16 KiB), graph construction/reverse adjacency excluded from prep timer. No GPU measured.

| Budget | BFS solved | Landmark solved | BFS checks total | Landmark checks total |
|---|---:|---:|---:|---:|
|64|128/2035|642/2035|126404|111280|
|256|516/2035|1591/2035|456608|269468|
|1024|2035/2035|2035/2035|1031604|351676|

Total measured search times in ns: see planning_results.json. At full budget BFS 387.197 ms vs landmark 251.920 ms for entire pooled panel, excluding preparation; adding 32 mean preparations gives ~387.2 ms vs ~387.2 ms (break-even on this workload; small timing variations matter). At lower budgets landmark improves success but increases query latency. No claim of overall speed win including prep at this query mix.

## Reproduction

`python grounding.py` (PyTorch, NumPy; single CPU thread; overwrites results/checkpoints); `python planning.py` (NumPy); `node reward.mjs train.csv test.csv` (official unmodified CSV; blob hash verified). `results.json`, `reward_results.json`, `planning_results.json` hold per-seed results. Results are exploratory; no qualification without disjoint replication, prospective gates, authentic language-to-environment interaction, and tuned end-to-end neural controls.