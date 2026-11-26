# COG-UNIFIED-019 — competent-baseline associative computation study
2026-11-26. CPU developmental experiments; recovered UNIFIED-018 reference COG and associative mechanisms. 105 training executions, 93 independent lineages, 12 optimizer-preserving continuations; 57,600 optimizer updates counted once, 1,651,200 synthetic history presentations, 1,536,000 language-byte presentations. 105/105 checkpoint hash/evaluation replays. No official campaign credit, no canonical promotion, no accelerator or protected TEST.

## Basic retrieval — 18 fits
One 32D layer, approx 12K parameters, 400 updates batch32; 8 distinct random keys of 16, 8 random values, four queries. COG has 256 logical FP32 bytes at eight-record prefix; narrow attention has 256 cache bytes then grows. Three seeds, 256 fresh histories per seed. Direct associative readout replaces shared nonlinear bank. Joint model uses learned writes with no hard query-HOLD for auto and squared arms.

|Arm|Individual %|Complete %|
|---|---:|---:|
|Native COG|55.18|5.60|
|Compact delta explicit HOLD|100|100|
|Compact delta learned sigmoid|100|100|
|Compact delta squared sigmoid|100|100|
|Compact delta straight-through binary|83.30|61.33|
|Narrow attention|100|100|

At 16 records OOD complete accuracy: auto 95.57%, squared 99.09%, explicit HOLD 99.87%, attention 100%. Gate means record/query: auto .704693/.047438; squared .718804/.019467. Cross-identity cosine RMS .028659 auto, .029249 squared, .031165 HOLD, .082817 binary. All nearest-key checks pass, which does not imply value custody. No exact noninterference claim for learned soft gates.

## Simultaneous retrieval and aggregation — 36 fits
Single query position produces two values and sum mod8, same input information and loss. Initial merged-query representation, 18 fits: one-head narrow SDPA 36.72% complete pairs, 90.89% aggregate; two-head narrow and four-head full SDPA 100% both; associative one-read 10.16% pairs, 18.88% aggregate; two-read 17.19% and 38.41%; two-read auto 19.14% and 40.62%. This is a qualified primitive deficit, not a general LLM deficit.

Secondary factorized query experiment, 18 fits, three new seeds and separate fresh panels, 600 updates. Query components are explicitly separated and both controls receive them. Tied learned address projections for separate keys, shared 16x8 matrix plus 32D working state, 640 logical bytes, narrow SDPA prefix cache matched at eight records. No learned semantic query decomposition claimed.

|Arm|Both values correct %|Aggregate %|
|---|---:|---:|
|Factorized two-head SDPA|100|100|
|Factorized single read|21.22|72.53|
|Additive HOLD|99.87|98.96|
|Additive learned writes|69.92|73.44|
|Delta HOLD|99.87|99.48|
|Delta learned writes|100|99.74|

Delta learned writes: 768/768 both values, 766/768 aggregate. Fresh intervention panel: complete 100%, same-address 11.20%, swapped read heads 11.20%, uniform 2.34%, zero memory 1.56%. OOD 16-record complete 99.35%. Both pathways necessary on tested task; explicit decomposition limits claim.

## Unpacked key-value retrieval/induction — 12 fits and 12 continuations, plus 12 secondary fits
Original two-layer alternating key/value token sequences with positional inputs: at 400 updates individual accuracy SDPA 35.74%, native 36.62%, additive 35.77%, delta 35.42%. After 1600 updates SDPA 37.01%, native 36.52%, additive 37.43%, delta 37.27%. Baseline competence fails; do not reject COG for this panel.
Secondary shared trainable two-token mixing layer given equally to all arms, new seeds, 600 updates. Complete four-query accuracy (6 records / 12 OOD): SDPA 100% /95.70%; native .13% /.52%; additive 94.79% /68.75%; delta 97.66% /51.56%. Delta misses every-seed near-attention gate. Engineered local context, not spontaneous induction head discovery.

## Language integration — 15 fits
Two 64D layers ~150K parameters, 400 updates, 102400 byte exposures per fit; same development-exposed Usenet corpus, document-held windows exact-match-filtered. Raw corpus omitted from public archive. No adequate scaling or independent corpus claim. 
|Arm|Dev BPB|Test BPB|Recurrent state bytes|
|---|---:|---:|---:|
|Native64|3.7301406421|3.8100946746|1024|
|Native128|3.7224693921|3.8130297544|1536|
|Additive|3.7724917160|3.8547370071|1536|
|Delta|3.7203792764|3.8059204116|1536|
|RoPE SDPA|3.7699080624|3.8611368595|growing|
Delta fails 1%-gain-every-seed gate vs state-matched native. Native128 same-target 32/128/256-context BPB 3.6499/3.6177/3.6178; delta 3.6503/3.6472/3.6931; add 3.7328/3.9353/4.3128. Associative primitive gains do not establish language-model superiority.

## Execution and reproducibility
Delta state update is affine: A=I-beta*k*k^T and B=beta*k*v^T, composition (A2*A1,A2*B1+B2). Associative dense matrix scan vs FP64 sequential max observed error 3.55e-6, gradient deviation 1.49e-8. CPU matrix width16, T512, B1: sequential 11.993ms, scan 3.625ms; B8: sequential 15.854ms, scan 36.447ms. Shape-dependent implementation overhead; no end-to-end speed, quality-matched economics or GPU claim. All 105 evaluations exactly replayed, zero tested prediction continuation disagreements, max split logit deviation 4.77e-6. Local module continuation includes previous token and correct positional frame; counter not included in logical state. No universal proof of learned recall.

## Evidence and custody
Priority ZIP committed with 9 lossless Safetensors inference exports (3 compact auto, 3 compact squared, 3 factorized delta-auto), exact source, protocols, all raw numerical fit results, verification and manifest. No raw corpus or optimizer bytes in public ZIP. Priority ZIP SHA256 354621ddbcaf3b6e814a1e5c3deb985dc7591f77005951f40822f59a87981d0e, bytes 546533, git blob 738814e41b2266143da10ff465195f0824cfea9d.
Local full original evidence ZIP 55744763 bytes SHA256 6639caa9909c47e2b3618136612ecadce088b68501c3d52e5b9d98fc4fe66086. Local complete inference release ZIP (105 exports) 13700902 bytes SHA256 c7757f95fd151dab8beb39d913bdec184a154ef3a9376b649f6b7b94d6dd0560. These hashes do not establish remote full-archive custody. See remote verification record. Historical backlog unchanged.

Disposition: compact and factorized associative retrieval are developmental bounded candidates. Basic recall and two-source aggregation near-attention on competent tests; general latent query decomposition, native induction, sustained language competitiveness and matched-quality lifecycle efficiency remain open. No canonical promotion.
