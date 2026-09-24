"""Run PNDS-GATE-001 Stage 2 and emit a JSON artifact."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .stage2_bandit import run

def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--train-episodes",type=int,default=2000)
    p.add_argument("--test-episodes",type=int,default=300)
    p.add_argument("--candidates",type=int,default=8)
    p.add_argument("--dim",type=int,default=8)
    p.add_argument("--noise",type=float,default=0.10)
    p.add_argument("--output",type=Path,default=Path("pnds_gate_001_stage2.json"))
    a=p.parse_args()
    result=run(a.train_episodes,a.test_episodes,a.candidates,a.dim,a.noise)
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(f"test_success={result['test_success_mean']:.4f}")
    print(f"test_intervention={result['test_intervention_effect_mean']:.4f}")
    print(f"artifact={a.output}")

if __name__=="__main__":
    main()
