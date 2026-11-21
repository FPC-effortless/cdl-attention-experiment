"""Generate reproducible stratified puzzle challenges and independent reference checks.

This is not COG-ATTN evaluation. Solutions and effort metrics are NEVER included in
public challenge inputs. For chess > mate-in-one use adversarial, all-reply proofs.
"""
from __future__ import annotations
import argparse, hashlib, json, statistics, sys
from collections import Counter,defaultdict
from pathlib import Path
from time import perf_counter_ns
from difficulty import (LEVELS,MAZE_SIZES,SUDOKU_SPEC,SLIDING_DEPTHS,HANOI_DISCS,CHESS_MATES,
  chess_pool,exact_mate_depth,make_sudoku_challenge,make_maze_challenge,maze_changed,
  sliding_distance_buckets,make_slide_exact,hanoi_moves,hanoi_min_moves,validate_hanoi_prefix,
  make_sliding4,encode_puzzle,digest)
from puzzles import ChessEndgame,Hanoi

SEEDS={'development':(8701,8707),'reserved':(9701,9707)}
DOMAINS=('chess','sudoku','maze','sliding3','hanoi')


def assess_chess(p,required):
    """Forcing criterion: a legal White move wins against EVERY Black reply."""
    assert exact_mate_depth(p,4)==required
    eligible=0
    from difficulty import force_mate
    for _,_,_,nextpos in p.white_moves():
        if nextpos.mate():eligible+=1;continue
        if required==1:continue
        replies=nextpos.black_replies()
        if replies and all(dest!=nextpos.heavy and
            force_mate(nextpos.wk,nextpos.heavy,dest,p.kind,required-1) for dest in replies):eligible+=1
    assert eligible>0
    return dict(scope='restricted_KQK_KRK_only',mate_in_white_moves=required,
                legal_first_moves=len(p.white_moves()),forced_winning_first_moves=eligible,
                verified_against_all_black_replies=True,full_chess_rules=False)


def build(output,rows_per_tier_seed=1):
    assert rows_per_tier_seed==1,'fixed protected split for the first pilot'
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    start=perf_counter_ns()
    slider=sliding_distance_buckets()
    chess={'Q':chess_pool('Q',target_each=6),'R':chess_pool('R',target_each=6)}
    chess_next=defaultdict(int);seen=set();public=[];secret=[]
    failures=[]
    for panel,seeds in SEEDS.items():
        for sidx,seed in enumerate(seeds):
            for level,title in enumerate(LEVELS,1):
                for domain in DOMAINS:
                    if __import__('os').environ.get('DEBUG_LADDER'):print('PROGRESS',panel,seed,level,domain,flush=True)
                    id=f'{panel}/{seed}/{domain}/L{level:02d}'
                    sub_seed=seed*101003+level*131+sum(map(ord,domain))*19
                    if domain=='chess':
                        kind=('Q' if (sidx+level+(panel=='reserved'))%2==0 else 'R')
                        depth=CHESS_MATES[level-1];ix=chess_next[depth,kind];chess_next[depth,kind]+=1
                        wk,heavy,bk=chess[kind][depth][ix]
                        p=ChessEndgame(wk,heavy,bk,kind)
                        measure=assess_chess(p,depth)
                        meta=dict(kind=kind,maximum_required_mate_in=depth,opponent='perfect_adversarial_black')
                    elif domain=='sudoku':
                        p,measure=make_sudoku_challenge(sub_seed,level)
                        # Exact uniqueness, independent from the difficulty proxy.
                        from difficulty import count_sudoku_solutions_capped
                        count,_=count_sudoku_solutions_capped(p,3000)
                        assert count==1
                        meta=dict(size=p.n,requested_removed=SUDOKU_SPEC[level-1][1],unique_solution=True)
                    elif domain=='maze':
                        p,ref=make_maze_challenge(sub_seed,level)
                        measure=dict(shortest_path_moves=len(ref.path),bfs_expanded=ref.expanded,
                                     verified_replay=p.apply(ref.path))
                        meta=dict(size=p.size,fully_observed=True)
                        if level>=8:
                            shifted,r=maze_changed(p,ref.path)
                            if shifted is not None:
                                assert r.path and shifted.apply(r.path) and not shifted.apply(ref.path)
                                measure.update(shifted=True,shift_replanned_steps=len(r.path),shift_expanded=r.expanded,
                                               invalid_old_path=True)
                                meta['shifted_input']=encode_puzzle(shifted)
                            else:
                                measure.update(shifted=False)
                                failures.append({'id':id,'cause':'No valid shifted obstacle with alternate route; dynamic challenge excluded'})
                    elif domain=='sliding3':
                        if level==10 and sidx==1:
                            failures.append({'id':id,'cause':'Only two 31-move 8-puzzle states exist; exclude instead of leaking duplicates across splits'})
                            continue
                        depth=SLIDING_DEPTHS[level-1]
                        # Avoid duplicate within same depth, especially at 31 (only two board states).
                        try_index=0
                        while True:
                            p=make_slide_exact(sub_seed+try_index*71171,depth,slider)
                            key=digest((domain,encode_puzzle(p)))
                            if key not in seen:break
                            try_index+=1
                            if try_index>300:raise RuntimeError(f'No distinct 8-puzzle board depth {depth}')
                        ref=p.shortest()
                        assert p.apply(ref.path) and len(ref.path)==depth
                        measure=dict(exact_optimal_moves=depth,astar_expanded=ref.expanded,
                                     entire_reachable_state_space_certified=181440)
                        meta=dict(size=3,exact_optimal_depth=True)
                    elif domain=='hanoi':
                        discs=HANOI_DISCS[level-1];p=Hanoi(discs);minimum=hanoi_min_moves(discs)
                        prefix=[]
                        for i,move in enumerate(hanoi_moves(discs)):
                            if i==min(minimum,100):break
                            prefix.append(move)
                        length,state=validate_hanoi_prefix(discs,prefix)
                        assert length==min(minimum,100)
                        full_check=None
                        if discs<=12:
                            path=list(hanoi_moves(discs))
                            assert len(path)==minimum and p.apply(path)
                            full_check=True
                        measure=dict(exact_optimal_moves=minimum,checked_prefix_moves=length,
                                     full_trace_executed=full_check,
                                     symbolic_minimum_certified=True)
                        meta=dict(discs=discs,optimality_proved_by_standard_hanoi_recurrence=True,
                                  action_trace_unrolled=bool(full_check))
                    else:raise ValueError(domain)
                    visible={'id':id,'split':panel,'level':level,'label':title,'domain':domain,
                             'input':encode_puzzle(p),'rules':meta}
                    fingerprint=digest((domain,visible['input']))
                    if domain!='hanoi':
                        assert fingerprint not in seen,(id,'duplicate')
                        seen.add(fingerprint)
                    visible['input_sha256']=fingerprint
                    public.append(visible)
                    secret.append({'id':id,'level':level,'domain':domain,'split':panel,'reference':measure,
                                   'input_sha256':fingerprint})
                # Additional frontier family. This is scramble-certified, not shortest-distance-certified.
                if level>=8:
                    scrambles={8:24,9:48,10:96}[level]
                    p,witness=make_sliding4(sub_seed+88397,scrambles)
                    assert len(witness)==scrambles and p.apply(witness)
                    id=f'{panel}/{seed}/sliding4/L{level:02d}'
                    fingerprint=digest(('sliding4',p.start))
                    if fingerprint in seen:
                        failures.append({'id':id,'cause':'duplicate 15 puzzle generated; candidate excluded'})
                    else:
                        seen.add(fingerprint)
                        public.append(dict(id=id,split=panel,level=level,label=title,
                                           domain='sliding4',input=encode_puzzle(p),input_sha256=fingerprint,
                                           rules={'scramble_walk_moves':scrambles,'exact_optimal_depth':False,
                                                  'solvable_by_reversal':True}))
                        secret.append(dict(id=id,level=level,domain='sliding4',split=panel,
                                           reference={'verified_upper_bound_moves':scrambles,'exact_optimal_moves':None,
                                                      'full_witness_replay_verified':True},input_sha256=fingerprint))
    assert len(public)==len(secret) and len(set(x['id'] for x in public))==len(public)
    with (output/'cases.jsonl').open('w') as f:
        for r in public:f.write(json.dumps(r,separators=(',',':'),sort_keys=True)+'\n')
    with (output/'reference_metrics.jsonl').open('w') as f:
        for r in secret:f.write(json.dumps(r,separators=(',',':'),sort_keys=True)+'\n')
    family=Counter(x['domain'] for x in public)
    by_level={str(k):dict(label=LEVELS[k-1],n=sum(r['level']==k for r in public),
                           domains=dict(Counter(r['domain'] for r in public if r['level']==k))) for k in range(1,11)}
    summary={'status':'REFERENCE ENVIRONMENTS ONLY — NO COG-ATTN POLICY WAS EVALUATED',
        'study':'COG-ATTN-087–090 puzzle difficulty ladder',
        'levels':list(LEVELS),'seeds':SEEDS,'total':len(public),'families':dict(family),
        'level_counts':by_level,'known_validation_gaps':failures,
        'chess_limit':'KQK/KRK forced mate in 1–4 White moves; not full chess/engine-rated difficulty',
        'sudoku_limit':'MRV/guess and clue metrics are proxies, not a human-calibrated difficulty rating',
        'hanoi_limit':'large-disc counts certified symbolically, full traces intentionally not executed',
        'sliding4_limit':'reversible scrambles but exact minimum unknown',
        'held_out_split':'seed-disjoint and input-disjoint except canonical Hanoi repetitions',
        'wall_ns':perf_counter_ns()-start}
    (output/'summary.json').write_text(json.dumps(summary,sort_keys=True,indent=2)+'\n')
    return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default=str(Path(__file__).parent/'results'))
    args=parser.parse_args();print(json.dumps(build(args.output),indent=2))
