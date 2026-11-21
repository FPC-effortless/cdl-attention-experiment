"""COG-098 readiness gate for the actual single-agent end-to-end architecture.

Only reports whether full-system source and authentic test receipts exist.
Does not substitute a synthetic baseline or hallucinate experiment scores.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
METHODS = ('observe', 'propose', 'act', 'learn', 'freeze', 'snapshot', 'restore', 'explain_provenance')
SOURCES = ('research/cog-071-080/tracker.py', 'research/cog-071-080/boundary.py',
           'research/cog-071-080/plans.py', 'research/cog-071-080/causal.py',
           'research/cog-097e/experiment.py')
DOMAINS = ('software', 'dialogue', 'interactive')


def methods_of(path):
    if not path.exists():
        return set()
    node = ast.parse(path.read_text())
    for item in node.body:
        if isinstance(item, ast.ClassDef) and item.name == 'FullCOGAgent':
            return {m.name for m in item.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))}
    return set()


def audit():
    sources = {}
    for name in SOURCES:
        p = ROOT / name
        sources[name] = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
    has = methods_of(HERE/'agent.py')
    components = {
      'all_upstream_sources': all(sources.values()),
      'single_full_agent_interface': all(x in has for x in METHODS),
      'training_entrypoint': (HERE/'train.py').exists(),
      'frozen_test_entrypoint': (HERE/'evaluate.py').exists(),
      'authentic_data_manifest': False,
      'authentic_holdout_receipts': False,
    }
    mp = HERE/'data_manifest.json'
    if mp.exists():
        try:
            m = json.loads(mp.read_text())['benchmarks']
            components['authentic_data_manifest'] = all(m.get(d,{}).get('source_sha256') and m[d].get('split_sha256') for d in DOMAINS)
        except (KeyError, ValueError, TypeError):
            pass
    rp = HERE/'qualification_results.json'
    if rp.exists():
        try:
            r = json.loads(rp.read_text())
            components['authentic_holdout_receipts'] = bool(r.get('frozen_checkpoint_sha256')) and all(
                r.get('benchmarks',{}).get(d,{}).get('evaluation_receipt_sha256') and
                r['benchmarks'][d].get('n_tasks',0)>0 for d in DOMAINS)
        except (KeyError, ValueError, TypeError):
            pass
    ok = all(components.values())
    return {'experiment':'COG-ATTN-098','scientific_decision':'READY_FOR_EXTERNAL_EVALUATION' if ok else 'NOT_QUALIFIED',
            'gate_type':'source_and_receipt_readiness_not_model_quality','components':components,
            'missing_agent_methods':sorted(set(METHODS)-has),'source_sha256':sources,
            'unmet_gates':[k for k,v in components.items() if not v]}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',default=str(HERE/'readiness.json'))
    parser.add_argument('--strict',action='store_true')
    args=parser.parse_args()
    result=audit()
    Path(args.out).write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,indent=2))
    return 2 if args.strict and result['scientific_decision']=='NOT_QUALIFIED' else 0


if __name__=='__main__':
    raise SystemExit(main())
