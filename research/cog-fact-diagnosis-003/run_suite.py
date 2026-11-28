"""Run the frozen-checkpoint diagnostic suite from a checkout of the research branch."""
import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair-dir", required=True, type=Path, help="Extracted ARM-balanced-REGIME-SEED/weights.pt folders")
    parser.add_argument("--scale-dir", required=True, type=Path, help="Folder of original ARM/scale/seed archives named as in the results")
    parser.add_argument("--corpus", required=True, type=Path, help="Pinned BALANCE corpus directory")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parent
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    def run(script, spec, label, *extra):
        cmd = [sys.executable, here / (script+".py"), spec, args.out / (label+".json"), *extra]
        subprocess.run(cmd, check=True, env=env)
    for seed in (9731, 9743, 9769):
        for arm in ("C", "GQA", "GRU", "R"):
            regimes = ("final", "trace", "pair", "trace_pair") if arm == "C" else ("trace_pair",)
            for regime in regimes:
                label = f"{arm}-balanced-{regime}-{seed}"
                spec = f"{args.repair_dir / label / 'weights.pt'}:{arm}:1m:{seed}"
                if arm != "R":
                    run("binding", spec, "binding-"+label)
                if regime != "trace_pair":
                    continue
                run("trace_fault", spec, f"{arm}-{seed}", *(["--english"] if arm=="C" else []))
                if arm == "C":
                    run("binding", spec, f"english-independent-{seed}", "english")
                    run("query_path", spec, f"query-{seed}")
                    run("write_site", spec, f"write-{seed}")
                    run("persistence", spec, f"persistence-{seed}")
                    run("language", spec, f"language-{seed}", args.corpus)
                    run("objectives", spec, f"grad-{seed}", args.corpus)
    for arm, seed in (("A",9731),("A",9743),("GQA",9743),("GRU",9743),("C",9769)):
        name = f"1m-{arm}-{seed}"
        spec = f"{args.scale_dir / (name+'.zip')}:{arm}:1m:{seed}"
        for domain in ("symbol", "english"):
            run("binding", spec, f"parent-{name}-{domain}", domain)
    for scale in ("1m", "10m"):
        spec = f"{args.scale_dir / (scale+'-C-9769.zip')}:C:{scale}:9769"
        run("trace_fault", spec, scale+"-orig", *(["--english"] if scale=="1m" else []))
        run("language", spec, "language-"+scale, args.corpus)
    spec = "random:C:1m:9769"
    run("trace_fault", spec, "random", "--english")
    run("persistence", spec, "persistence-random")
    run("binding", spec, "binding-random")
    run("binding", spec, "english-independent-random", "english")


if __name__ == "__main__":
    main()
