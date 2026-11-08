"""Compile baseline/axioms first; reject specific semantic mutations afterwards.

No failed baseline, missing compiler, syntax error or missing import counts as a
successful mutation test. This samples fault sensitivity, not all non-vacuity.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'formal/relational/lean'
MODULES = ('Basic', 'Paged', 'Programs', 'Mixture', 'Audit')


def compile_sources(lean, directory):
    env = dict(os.environ, LEAN_PATH=str(directory))
    logs = []
    for name in MODULES + ('Axioms',):
        source = directory / 'Relational' / (name + '.lean')
        run = subprocess.run([lean, '-o', str(source.with_suffix('.olean')), str(source)],
                             cwd=directory, env=env, text=True, capture_output=True, timeout=90)
        logs.append(dict(module=name, exit_code=run.returncode,
                         output=run.stdout + run.stderr))
        if run.returncode:
            return False, logs
    return True, logs


def prepare(directory):
    shutil.copytree(SOURCE, directory)
    names = []
    for module in MODULES:
        text = (directory / 'Relational' / (module + '.lean')).read_text()
        names += re.findall(r'^theorem (\w+)', text, flags=re.M)
    (directory / 'Relational/Axioms.lean').write_text('import Relational.Audit\n' +
        '\n'.join('#print axioms Relational.' + n for n in names) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lean', default='lean')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = dict(source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in SOURCE.rglob('*.lean')}, mutations=[])
    try:
        version = subprocess.run([args.lean, '--version'], text=True, capture_output=True, timeout=20)
        report['compiler'] = version.stdout + version.stderr
        if version.returncode or 'version 4.15.0' not in report['compiler']:
            raise RuntimeError('pinned Lean 4.15.0 did not start')
        with tempfile.TemporaryDirectory(prefix='lrc-lean-audit-') as temp:
            base = Path(temp) / 'baseline'
            prepare(base)
            ok, logs = compile_sources(args.lean, base)
            report['baseline'] = dict(passed=ok, logs=logs)
            if not ok:
                print(logs[-1]['output'], flush=True)
                raise RuntimeError('baseline compilation failed; mutations not attempted')
            axiom_text = logs[-1]['output']
            requested = re.findall(r'#print axioms (\S+)',
                                   (base / 'Relational/Axioms.lean').read_text())
            if not requested or any(name not in axiom_text for name in requested):
                raise RuntimeError('missing theorem axiom output')
            allowed = {'propext', 'Classical.choice', 'Quot.sound'}
            for group in re.findall(r'depends on axioms: \[([^\]]*)\]', axiom_text):
                if not set(x.strip() for x in group.split(',') if x.strip()) <= allowed:
                    raise RuntimeError('unapproved axiom dependency')
            if 'sorryAx' in axiom_text:
                raise RuntimeError('incomplete proof dependency')
            report['axiom_audit_passed'] = True
            mutations = [
                ('strict_causality', 'Basic', 'w.position < t', 'w.position ≤ t'),
                ('page_offset', 'Paged', '(page * budget)', 'page'),
                ('mixture_wrong_weight', 'Mixture', 'b * cell.2', 'a * cell.2'),
            ]
            for name, module, before, after in mutations:
                directory = Path(temp) / name
                prepare(directory)
                target = directory / 'Relational' / (module + '.lean')
                source = target.read_text()
                if before not in source:
                    raise RuntimeError('mutation target missing')
                target.write_text(source.replace(before, after))
                ok, logs = compile_sources(args.lean, directory)
                failure = logs[-1]['output']
                expected = any(s in failure for s in ('type mismatch', 'unsolved goals',
                                                       "tactic 'decide'", 'application type mismatch',
                                                       'omega could not prove the goal'))
                expected_module = 'Audit' if name == 'page_offset' else module
                killed = not ok and expected and logs[-1]['module'] == expected_module
                report['mutations'].append(dict(name=name, killed=killed, logs=logs))
                print(json.dumps(dict(mutation=name, killed=killed,
                                      failing_module=logs[-1]['module'])), flush=True)
                if not killed:
                    print(failure, flush=True)
                    raise RuntimeError('mutation survived or failed for an unrecognized reason')
        report['status'] = 'PASS'
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        report['status'] = 'BLOCKED_OR_FAILED'
        report['error'] = str(error)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('status', 'compiler', 'error') if k in report}))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
