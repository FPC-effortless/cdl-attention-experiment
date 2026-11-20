"""COG-066 bridge safety, provenance and cohort invariants."""
import json
from pathlib import Path
from bridge.cog066_source import acquire
from bridge.experiment_086 import learn, exact_search, scramble_options, DOMAINS
from evaluate_submissions import from_case
from puzzles import make_maze, shift_maze, Sliding, Hanoi

HERE=Path(__file__).resolve().parents[1]

def test_acquisition_needs_three_independent_families():
    repetitive=[dict(verified=True,path=['R','R','D','D']*10,source_family='same')] * 30
    assert acquire(repetitive)==()
    diversified=[dict(verified=True,path=['R','R','D','D']*4,source_family=str(i)) for i in range(3)]
    assert acquire(diversified)

def test_skill_transfer_never_uses_replication_labels():
    cases=[json.loads(x) for x in (HERE/'puzzle_cases.jsonl').read_text().splitlines()]
    opts,setup=learn(cases)
    assert set(opts)==set(DOMAINS)
    assert all(n==12 for n in setup['certified'].values())
    training={c['environment_digest'] for c in cases if c['split']=='primary' and c['domain'] not in ('hanoi_3','hanoi_5')}
    held={c['environment_digest'] for c in cases if c['split']=='replication' and c['domain'] not in ('hanoi_3','hanoi_5')}
    assert not (training & held)
    assert opts['hanoi_3']==opts['hanoi_5']==()
    scrambled=scramble_options(opts)
    for d in DOMAINS:
        assert [len(s) for s in opts[d]]==[len(s) for s in scrambled[d]]

def test_invalid_options_are_rejected_transactionally():
    p=make_maze(12345,11)
    normal=exact_search(p)
    # Broken shortcut must not change the state; exact primitives remain available.
    corrupted=exact_search(p,[('X','X'),('D','X'),('U','U','X')])
    assert normal['solved'] and corrupted['solved']
    assert len(normal['path'])==len(corrupted['path'])
    assert corrupted['invalid_options']>0

def test_mazes_never_execute_outdated_plan():
    count=0
    for seed in range(35):
        p=make_maze(seed,11,0.30)
        changed=shift_maze(p)
        if changed is None:continue
        old=exact_search(p)
        assert old['solved'] and not changed.apply(old['path'])
        repaired=exact_search(changed)
        assert repaired['solved'] and changed.apply(repaired['path'])
        count+=1
    assert count>=5

def test_all_held_arms_same_competence_and_no_optimism():
    f=HERE/'bridge'/'086-results.json'
    if not f.exists():return
    r=json.loads(f.read_text())
    assert all(not x['admitted'] for x in r['calibration'].values())
    assert set(r['totals'])=={'primitive','cog066','shuffled','gated'}
    assert all(x['solved']==x['n']==126 for x in r['totals'].values())
    assert r['totals']['primitive']['attempts']==r['totals']['gated']['attempts']
    assert r['totals']['cog066']['attempts']>r['totals']['primitive']['attempts']
    assert r['totals']['cog066']['popped']==r['totals']['primitive']['popped']
