"""Deterministic pilot for puzzle adapters and honest reference baselines.

USAGE: python run_pilot.py --out results.json
No COG-ATTN model is loaded or evaluated in this runner.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import hashlib,json,random,statistics,sys
from dataclasses import dataclass
from time import perf_counter_ns
from pathlib import Path
from puzzles import *

SEEDS={'primary':(8101,8107,8113),'replication':(9101,9107,9113)}
DOMAINS=('chess_Q','chess_R','sudoku_4','sudoku_9','maze_7','maze_11','maze_shift','hanoi_3','hanoi_5','slide_8','slide_14')

def tag(o):
    return hashlib.sha256(repr(o).encode('utf8')).hexdigest()

def generate(domain,seed,trial,chs):
    key=seed*1000+trial*17+sum(map(ord,domain))
    if domain=='chess_Q':return random.Random(key).choice(chs['Q'])
    if domain=='chess_R':return random.Random(key).choice(chs['R'])
    if domain=='sudoku_4':return make_sudoku(key,4,9)
    if domain=='sudoku_9':return make_sudoku(key,9,49)
    if domain=='maze_7':return make_maze(key,7)
    if domain=='maze_11':return make_maze(key,11)
    if domain=='maze_shift':
        for i in range(100):
            init=make_maze(key+i*191,11,0.30)
            shifted=shift_maze(init)
            if shifted:return (init,shifted)
        raise AssertionError('Failed to find a solvable changed maze')
    if domain=='hanoi_3':return Hanoi(3)
    if domain=='hanoi_5':return Hanoi(5)
    if domain=='slide_8':return make_sliding(key,8)
    if domain=='slide_14':return make_sliding(key,14)
    raise ValueError(domain)

def payload(p):
    if isinstance(p,Maze):return {'n':p.size,'walls':[[r,c] for r in range(p.size) for c in range(p.size) if (r,c) not in p.open_cells],'start':list(p.start),'goal':list(p.target)}
    if isinstance(p,Sudoku):return {'n':p.n,'grid':[list(p.board[i:i+p.n]) for i in range(0,len(p.board),p.n)]}
    if isinstance(p,Hanoi):return {'discs':p.discs,'state':list(p.start),'goal':list(p.target)}
    if isinstance(p,Sliding):return {'grid':[list(p.start[i:i+3]) for i in (0,3,6)]}
    if isinstance(p,ChessEndgame):
        name=lambda i:'abcdefgh'[i%8]+str(i//8+1)
        board=['']*64;board[p.wk]='K';board[p.heavy]=p.kind;board[p.bk]='k'
        ranks=[]
        for r in reversed(range(8)):
            rank='';empty=0
            for x in range(8):
                pce=board[r*8+x]
                if not pce:empty+=1
                else:
                    if empty:rank+=str(empty);empty=0
                    rank+=pce
            if empty:rank+=str(empty)
            ranks.append(rank)
        fen='/'.join(ranks)+' w - - 0 1'
        return {'fen':fen,'pieces':{'white_king':name(p.wk),'white_'+('queen' if p.kind=='Q' else 'rook'):name(p.heavy),'black_king':name(p.bk)},'objective':'Find a legal mate in one','scope':'K+Q/R vs K only'}
    raise TypeError(type(p))

def run(output, trials=6):
    t0=perf_counter_ns();chs={k:chess_candidates(k) for k in ('Q','R')}
    rows=[];public=[];shift_rows=[];seen_inputs={}
    # Actual exact experience store: exact puzzle digest => verified plan.
    # This is a tabular cache reference, not an intelligent learning model.
    for panel,seeds in SEEDS.items():
        for seed in seeds:
            order=[(domain,i) for domain in DOMAINS for i in range(trials)]
            random.Random(seed).shuffle(order)
            for domain,i in order:
                # Reject accidental duplicates across primary/replication for non-Hanoi tasks.
                # Canonical Hanoi 3/5 intentionally repeats; it is a procedural control.
                retry=0
                while True:
                    generated=generate(domain,seed,i+retry*1000,chs)
                    p,shift=(generated if domain=='maze_shift' else (generated,None))
                    candidate_digest=tag((domain,payload(p)))
                    if domain.startswith('hanoi_') or candidate_digest not in seen_inputs:break
                    retry+=1
                    if retry>200:raise AssertionError('could not construct a new puzzle')
                seen_inputs.setdefault(candidate_digest,(panel,seed,i))
                start=perf_counter_ns();solution=p.solve()[0] if isinstance(p,Sudoku) else p.shortest();exact_ns=perf_counter_ns()-start
                assert solution.path and p.apply(solution.path),(panel,seed,domain,i,'invalid oracle')
                naive=bool(p.greed())
                cache={} # An exact-match cache may learn only from THIS task, not another split.
                first_key=tag((domain,payload(p)))
                assert first_key not in cache,('unexpected duplicate fixture',panel,seed,domain,i)
                cache[first_key]=list(solution.path)
                # Repeat exact puzzle (simulates a learned exact plan), including lookup
                begin=perf_counter_ns();remembered=cache.get(tag((domain,payload(p))))
                reuse=remembered is not None and p.apply(remembered)
                cache_ns=perf_counter_ns()-begin
                assert reuse
                # Evaluate novel puzzle without including a trained support/fallback oracle.
                if shift is not None:
                    assert not shift.apply(solution.path)
                    changed_key=tag((domain,payload(shift)))
                    begin=perf_counter_ns();reuse_shift=cache.get(changed_key)
                    guard_hit=reuse_shift is not None and shift.apply(reuse_shift)
                    gated_ns=perf_counter_ns()-begin
                    assert not guard_hit
                    changed=shift.shortest();assert changed.path and shift.apply(changed.path)
                    # Counterfactual: an unsafe id-only cache would reuse stale invalid plan
                    shift_rows.append({'seed':seed,'trial':i,'old_length':len(solution.path),'new_length':len(changed.path),
                                       'unsafe_stale_invalid':not shift.apply(solution.path),'guard_rejected':not guard_hit,
                                       'repaired':shift.apply(changed.path),'guard_ns':gated_ns,'replan_expanded':changed.expanded})
                row={'panel':panel,'seed':seed,'domain':domain,'trial':i,'verified':True,'greedy_correct':naive,
                     'steps':len(solution.path),'expanded':solution.expanded,'exact_ns':exact_ns,
                     'cache_replay_ns':cache_ns,'repeat_cache_hit':reuse,'novel_cache_hit':False,'fixture_sha256':first_key,'generation_retry':retry}
                rows.append(row)
                public.append({'id':f'{panel}/{seed}/{domain}/{i}','domain':domain,'split':panel,
                               'input':payload(p),'environment_digest':first_key})
    summary={}
    for domain in DOMAINS:
        selected=[r for r in rows if r['domain']==domain]
        def pct(name):return round(100*sum(bool(r[name]) for r in selected)/len(selected),2)
        summary[domain]={'n':len(selected),'reference_solution_pct':pct('verified'),
                         'naive_pct':pct('greedy_correct'),
                         'exact_replay_hit_pct':pct('repeat_cache_hit'),
                         'novel_replay_hit_pct':pct('novel_cache_hit'),
                         'median_steps':statistics.median(r['steps'] for r in selected),
                         'median_expanded':statistics.median(r['expanded'] for r in selected),
                         'median_exact_ms':round(statistics.median(r['exact_ns'] for r in selected)/1e6,4),
                         'median_replay_ms':round(statistics.median(r['cache_replay_ns'] for r in selected)/1e6,4)}
    res={'benchmark':'COG-ATTN-081 puzzle environment pilot','run_type':'Puzzle environment and reference-control verification ONLY; NOT COG-ATTN model evaluation',
         'config':{'seeds':SEEDS,'trials_per_domain_per_seed':trials,'domains':DOMAINS,'external_dependencies':[]},
         'totals':{'puzzles':len(rows),'reference_valid':sum(r['verified'] for r in rows),
                   'naive_solved':sum(r['greedy_correct'] for r in rows),'repeated_cache_hits':sum(r['repeat_cache_hit'] for r in rows),
                   'novel_cache_hits':0,'unique_fixtures':len(seen_inputs),'intentional_hanoi_repeats':len(rows)-len(seen_inputs),'shifted_mazes':len(shift_rows),'invalid_stale_paths':sum(s['unsafe_stale_invalid'] for s in shift_rows),
                   'guard_rejections':sum(s['guard_rejected'] for s in shift_rows),'shift_repaired':sum(s['repaired'] for s in shift_rows)},
         'by_domain':summary,'rows':rows,'shifted':shift_rows,'wall_ns_including_fixture_generation':perf_counter_ns()-t0}
    output=Path(output);output.write_text(json.dumps(res,sort_keys=True,indent=2)+'\n')
    output.with_name('puzzle_cases.jsonl').write_text(''.join(json.dumps(p,sort_keys=True)+'\n' for p in public))
    return res

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',default='results.json');parser.add_argument('--trials',type=int,default=6);args=parser.parse_args()
    r=run(args.out,args.trials)
    print('Puzzles:',r['totals']['puzzles'],'valid reference:',r['totals']['reference_valid'])
    print('Controls are NOT the COG-ATTN architecture.')
    for k,v in r['by_domain'].items():print(f'{k:12} {v["n"]:3} valid={v["reference_solution_pct"]:5.1f}% naive={v["naive_pct"]:5.1f}% repeat-hit={v["exact_replay_hit_pct"]:5.1f}% unseen-hit={v["novel_replay_hit_pct"]:5.1f}%')
    print('Shift controls:',r['totals']['shifted_mazes'],'unsafe stale paths rejected:',r['totals']['guard_rejections'])
