# Frozen intervention diagnosis

192 paired evaluations; 12 frozen checkpoints; 24 exact no-op replays. Positive NLL delta means ablation hurts. Negative means it helps. DEV is exposed development data, never protected TEST.

| Arm | Panel | Intervention | Mean NLL change | Prediction changes |
|---|---|---|---:|---:|
| hard | fit | values | +0.1019 | 4.2% |
| hard | fit | regulation | +0.0503 | 1.6% |
| hard | fit | history | +3.5780 | 72.7% |
| hard | fit | links | +0.0044 | 0.8% |
| hard | fit | cursor | +0.0834 | 3.1% |
| hard | fit | permute | +0.6897 | 21.4% |
| hard | fit | all | +3.4937 | 74.7% |
| hard | dev | values | -0.0965 | 25.7% |
| hard | dev | regulation | -0.0478 | 18.4% |
| hard | dev | history | +0.0540 | 74.8% |
| hard | dev | links | -0.0368 | 13.0% |
| hard | dev | cursor | -0.0335 | 24.2% |
| hard | dev | permute | -0.0411 | 38.8% |
| hard | dev | all | -0.6037 | 78.8% |
| soft_routes | fit | values | +0.1827 | 4.9% |
| soft_routes | fit | regulation | +0.0091 | 1.3% |
| soft_routes | fit | history | +3.0764 | 66.9% |
| soft_routes | fit | links | +0.0000 | 0.0% |
| soft_routes | fit | cursor | +0.0567 | 2.9% |
| soft_routes | fit | permute | +3.1226 | 64.3% |
| soft_routes | fit | all | +3.6637 | 71.4% |
| soft_routes | dev | values | -0.0415 | 50.3% |
| soft_routes | dev | regulation | +0.0465 | 14.5% |
| soft_routes | dev | history | -0.1190 | 68.4% |
| soft_routes | dev | links | -0.0018 | 3.5% |
| soft_routes | dev | cursor | +0.0417 | 16.7% |
| soft_routes | dev | permute | +0.4293 | 55.5% |
| soft_routes | dev | all | -0.5373 | 79.6% |
| soft_admission | fit | values | +0.0589 | 2.1% |
| soft_admission | fit | regulation | +0.0078 | 1.0% |
| soft_admission | fit | history | +1.8697 | 54.7% |
| soft_admission | fit | links | +0.0008 | 0.0% |
| soft_admission | fit | cursor | +0.0528 | 4.4% |
| soft_admission | fit | permute | +0.1485 | 19.3% |
| soft_admission | fit | all | +1.9702 | 53.9% |
| soft_admission | dev | values | -0.0490 | 8.2% |
| soft_admission | dev | regulation | -0.0320 | 10.3% |
| soft_admission | dev | history | -0.1970 | 62.8% |
| soft_admission | dev | links | -0.0036 | 5.8% |
| soft_admission | dev | cursor | -0.0909 | 15.5% |
| soft_admission | dev | permute | +0.1117 | 31.0% |
| soft_admission | dev | all | -0.4846 | 64.4% |
| residual_front | fit | values | +0.0591 | 2.1% |
| residual_front | fit | regulation | +0.0144 | 1.6% |
| residual_front | fit | history | +2.4549 | 62.8% |
| residual_front | fit | links | +0.0002 | 0.3% |
| residual_front | fit | cursor | +0.0529 | 2.6% |
| residual_front | fit | permute | +0.1174 | 5.5% |
| residual_front | fit | all | +2.4715 | 62.0% |
| residual_front | dev | values | -0.1952 | 10.6% |
| residual_front | dev | regulation | -0.0448 | 12.0% |
| residual_front | dev | history | -0.5082 | 69.3% |
| residual_front | dev | links | +0.0065 | 3.1% |
| residual_front | dev | cursor | -0.0860 | 10.9% |
| residual_front | dev | permute | -0.2354 | 24.7% |
| residual_front | dev | all | -0.7273 | 72.4% |

Persistent ablations alter state distributions and cannot establish semantic correctness or identify the cause of the previous long-run collapse. Routing histograms, confidence, gates, per-seed results and KL are in results.json. No weights were updated.
