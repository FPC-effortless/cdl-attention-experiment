# Replay COG-CLAUSE-ORDER-001

Python 3.12.14 and CPU PyTorch 2.10.0+cpu were used. Set
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`.
All stages were invoked locally and are label-separated in code; this was
not a native TrainLab run or an externally independent secret custody test.
Use fresh paths to avoid overwriting results.

```sh
python run.py test
python run.py build --data data-v1
sha256sum -c LOCK.sha256
mkdir results
for seed in 3371 3471 3571; do
  for arm in sym order raw; do
    python run.py fit --data data-v1 --out results --seed "$seed" --arm "$arm"
  done
done
python run.py predict --data data-v1 --out results
python run.py judge --data data-v1 --out results
```

Fit reads only `train.json`; the predictor reads only held prompts and exact fitted
weights. The offline judge independently derives answer labels from raw keys
and requires the held truth wire to agree. Compare decisions and counts,
not the digest of a report with wall-clock fields or a PyTorch container
filename that can change across runtimes. `recount.py` verifies the original
wire custody and recomputes the aggregates without a model. The cohort has
now been exposed for research, so its reuse cannot yield a new blind claim.
