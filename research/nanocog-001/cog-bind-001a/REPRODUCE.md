# COG-BIND-001A replay

CPU, Python 3.12.14, PyTorch 2.10.0+cpu. Set `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`. Run in a clean directory with the source, protocol, and lock. This synthetic study's inputs are now exposed for replay, not a fresh prospective evaluation.

```sh
python run.py test
python run.py build --data data-v1
sha256sum -c LOCK.sha256
mkdir results
for seed in 5371 5471 5571; do
  for arm in answer support; do
    python run.py fit --data data-v1 --out results --arm "$arm" --seed "$seed"
  done
done
python run.py predict --data data-v1 --out results
python run.py judge --data data-v1 --out results
python gold_replay.py --data data-v1 --out results
```

The fit actor reads training/calibration; the held actor reads prompts only. The post-score gold replay is separate and uses exact saved query/value vectors. The independent recount uses raw text and the retained output wires, no trained weights. See its command help for fixed artifact hashes. The private replay archives provide byte-level evidence and a SHA-256 manifest. Exact timestamps, elapsed times and serialization may differ when rerun.
