# COG-MATCHED-BOUNDARY-001 replay

Use Python 3.12.14, PyTorch 2.10.0+cpu, single threads `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`. Run in an empty execution directory except for the source and protocol. Build is deterministic from the published synthetic generator; no pretrained weights or dataset download are required. All qualification inputs are now exposed as a development study, not a prospective test.

```sh
python run.py test
python run.py build --data data-v1
sha256sum -c LOCK.sha256
mkdir results
for seed in 4371 4471 4571; do
  for arm in dot add; do
    python run.py fit --data data-v1 --out results --seed "$seed" --arm "$arm"
  done
done
python run.py predict --data data-v1 --out results
python run.py judge --data data-v1 --out results
```

Fit reads only training and calibration, and prediction reads only the held text projection. Offline scoring uses the separate truth projection. The independent scorer reconstructs labels and supporting clause indices from the text. Compare scores and turn-to-turn predictions; wall-clock fields and serialization hashes may differ on replay. The private archive contains exact retained inputs, weights, logits, and clause weights, plus its SHA-256 manifest. CPU fit duration is not an inference benchmark.
