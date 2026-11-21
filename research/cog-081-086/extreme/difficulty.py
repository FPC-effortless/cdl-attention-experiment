"""COG-ATTN-087 puzzle curriculum: validated measurements, not model competence.

Completely deterministic standard-library-only tasks. Inference-time action policies
must not import this module because it contains reference solvers and hidden labels.
"""
from __future__ import annotations
from functools import lru_cache
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import hashlib
import random
import sys

PARENT=Path(__file__).resolve().parents[1]
if str(PARENT) not in sys.path:sys.path.insert(0,str(PARENT))
from puzzles import ChessEndgame, Maze, Sliding, Hanoi, Sudoku, make_sudoku, make_maze, adjacent

LEVELS=('Novice','Easy','Foundation','Intermediate','Challenging','Advanced','Expert','Master','Extreme','Frontier')
MAZE_SIZES=(7,9,11,15,21,31,45,65,95,151)
SUDOKU_SPEC=((4,4),(4,7),(4,9),(9,35),(9,42),(9,47),(9,50),(9,53),(9,55),(9,58))
SLIDING_DEPTHS=(2,4,6,8,12,16,20,24,28,31)
HANOI_DISCS=(2,3,4,5,7,9,12,15,18,22)
CHESS_MATES=(1,1,1,2,2,2,3,3,4,4)


def digest(obj):return hashlib.sha256(repr(obj).encode()).hexdigest()


def sliding_distance_buckets():
    """Prove 8-puzzle distances by BFS over its ENTIRE reachable state space."""
    goal=Sliding.target; queue=deque([goal]); dist={goal:0}; buckets={0:[goal]}
    while queue:
        state=queue.popleft();n=dist[state]+1
        for _,t in Sliding(state).neighbors(state):
            if t not in dist:
                dist[t]=n;queue.append(t);buckets.setdefault(n,[]).append(t)
    assert len(dist)==181440 and max(buckets)==31
    return buckets


def make_slide_exact(seed,depth,buckets):
    options=buckets[depth]
    # Stable selection independent of hash randomization.
    return Sliding(options[random.Random(seed).randrange(len(options))])


def count_sudoku_solutions_capped(p, max_nodes=3000):
    """Count at most two solutions, reject uncertain instances on bounded search."""
    nodes=0;solutions=0
    def rec(state):
        nonlocal nodes,solutions
        nodes+=1
        if nodes>max_nodes:raise RuntimeError('Uniqueness search budget exceeded')
        holes=[(p.choices(state,i),i) for i,v in enumerate(state) if not v]
        if not holes:
            solutions+=1;return
        opts,i=min(holes,key=lambda t:len(t[0]))
        for v in opts:
            if solutions>=2:return
            nxt=list(state);nxt[i]=v;rec(tuple(nxt))
    try:rec(p.board)
    except RuntimeError:return None,nodes
    return solutions,nodes


def sudoku_profile(sudoku,max_nodes=10000):
    """MRV+constraint accounting proxy, not human-rated difficulty.

    Inconclusive work always yields a lower bound, never a made-up exact cost.
    """
    s=tuple(sudoku.board);counts={'nodes':0,'guesses':0,'dead_ends':0,'forced':0}
    def recurse(state):
        counts['nodes']+=1
        if counts['nodes']>max_nodes:raise RuntimeError('Profiler budget exceeded')
        holes=[(sudoku.choices(state,i),i) for i,v in enumerate(state) if not v]
        if not holes:return True
        opts,i=min(holes,key=lambda p:len(p[0]))
        if len(opts)==0:counts['dead_ends']+=1;return False
        if len(opts)==1:counts['forced']+=1
        else:counts['guesses']+=1
        for value in opts:
            n=list(state);n[i]=value
            if recurse(tuple(n)):return True
        return False
    try:solved=recurse(s);counts['search_budget_exhausted']=False
    except RuntimeError:counts['search_budget_exhausted']=True;solved=None
    assert solved is True or counts['search_budget_exhausted']
    counts['clues']=sum(v!=0 for v in s);counts['blanks']=len(s)-counts['clues']
    counts['naked_singles_only']=bool(sudoku.greed())
    return counts


def make_sudoku_challenge(seed,level):
    """Generation uses uniquely certified clue deletion with bounded work."""
    n,remove=SUDOKU_SPEC[level-1]
    attempts=2 if n==9 else 1
    best=None;best_score=(-1,-1)
    for trial in range(attempts):
        base=make_sudoku(seed+trial*1000003,n,0)
        rng=random.Random(seed+trial*93371)
        board=list(base.board);order=list(range(n*n));rng.shuffle(order)
        verified_nodes=0
        for i in order:
            if board.count(0)>=remove:break
            previous=board[i];board[i]=0
            p=Sudoku(n,tuple(board))
            solutions,nodes=count_sudoku_solutions_capped(p,2000 if n==9 else 300)
            verified_nodes+=nodes
            if solutions!=1:board[i]=previous
        p=Sudoku(n,tuple(board));solutions,nodes=count_sudoku_solutions_capped(p,3000)
        assert solutions==1
        prof=sudoku_profile(p)
        prof['generation_uniqueness_nodes']=verified_nodes
        prof['requested_removed']=remove
        prof['achieved_requested_removal']=prof['blanks']==remove
        score=(prof['nodes']-prof['blanks'],prof['blanks'])
        if score>best_score:best=(p,prof);best_score=score
    return best


def hanoi_min_moves(discs):
    assert 1<=discs<=64
    return (1<<discs)-1


def hanoi_moves(discs,source=0,target=2,aux=1):
    """Streaming exact recursion, never expand 2**22 moves into RAM."""
    if discs==0:return
    yield from hanoi_moves(discs-1,source,aux,target)
    yield (source,target)
    yield from hanoi_moves(discs-1,aux,target,source)


def validate_hanoi_prefix(discs,prefix):
    state=tuple(0 for _ in range(discs));visited=0
    for step in prefix:
        choices=dict(Hanoi(discs).neighbors(state))
        assert step in choices
        state=choices[step];visited+=1
    return visited,state


@dataclass(frozen=True)
class Sliding4:
    start:tuple
    target=tuple(range(1,16))+(0,)
    def neighbors(self,s):
        i=s.index(0);r,c=divmod(i,4)
        for label,dr,dc in (('U',-1,0),('D',1,0),('L',0,-1),('R',0,1)):
            rr,cc=r+dr,c+dc
            if not(0<=rr<4 and 0<=cc<4):continue
            j=rr*4+cc;v=list(s);v[i],v[j]=v[j],v[i]
            yield label,tuple(v)
    def apply(self,path):
        s=self.start
        for move in path:
            nexts=dict(self.neighbors(s))
            if move not in nexts:return False
            s=nexts[move]
        return s==self.target


def make_sliding4(seed,steps):
    """A reversible scrambling witness, NOT proof of optimal solution depth."""
    rng=random.Random(seed);p=Sliding4(Sliding4.target);state=p.start;inverse=[];prev=None
    for _ in range(steps):
        choices=[(a,s) for a,s in p.neighbors(state) if s!=prev]
        a,after=rng.choice(choices);inverse.append({'U':'D','D':'U','L':'R','R':'L'}[a]);prev,state=state,after
    return Sliding4(state),tuple(reversed(inverse))


@lru_cache(maxsize=350000)
def force_mate(wk,heavy,bk,kind,remaining):
    """KQK/KRK ONLY: existential white move / universal black reply.

    Black capture of the rook/queen is a draw and a refutation. Black stalemate
    is not accepted. All black king replies are enumerated in current rules.
    """
    if remaining<1:return False
    p=ChessEndgame(wk,heavy,bk,kind)
    for _,_,_,child in p.white_moves():
        if child.mate():return True
        if remaining<=1:continue
        replies=child.black_replies()
        if not replies:continue
        if all(dest!=child.heavy and force_mate(child.wk,child.heavy,dest,kind,remaining-1) for dest in replies):
            return True
    return False


def exact_mate_depth(p,limit=4):
    for depth in range(1,limit+1):
        if force_mate(p.wk,p.heavy,p.bk,p.kind,depth):return depth
    return None


def chess_pool(kind, seed=8409, target_each=5):
    """Deterministic small curated endgame pool. Records exact *bounded* mate depth."""
    rng=random.Random(seed+sum(map(ord,kind)));pool={i:[] for i in range(1,5)}
    borders=[i for i in range(64) if i//8 in (0,7) or i%8 in (0,7)]
    seen=set()
    # Candidate construction must be kept off the agent's training input.
    for i in range(1500):
        bk=rng.choice(borders);wk=rng.randrange(64);heavy=rng.randrange(64)
        if len({wk,bk,heavy})<3 or adjacent(wk,bk):continue
        key=(wk,heavy,bk)
        if key in seen:continue
        seen.add(key)
        p=ChessEndgame(wk,heavy,bk,kind)
        if p.black_checked():continue
        if all(len(v)>=target_each for v in pool.values()):break
        # Avoid computing depth 4 when enough positions have already been found.
        depth=None
        for d in range(1,5):
            if all(len(pool[x])>=target_each for x in range(d,5)):break
            if force_mate(wk,heavy,bk,kind,d):depth=d;break
        if depth and len(pool[depth])<target_each:pool[depth].append(key)
    if any(len(v)<2 for v in pool.values()):
        raise RuntimeError('Could not find distinct certified endgame positions: '+str({k:len(v) for k,v in pool.items()}))
    return pool


def make_maze_challenge(seed,level):
    size=MAZE_SIZES[level-1];extra=max(.07,.20-(level-1)*.014)
    # Favor a longer verified path among different topologies; hardness metrics remain measured.
    candidates=[]
    for i in range(3):
        p=make_maze(seed+i*10007,size,extra=extra);r=p.shortest()
        assert r.path and p.apply(r.path)
        candidates.append((len(r.path),r.expanded,p,r))
    _,_,p,ref=max(candidates,key=lambda row:row[:2])
    return p,ref


def maze_changed(p,path):
    state=p.start
    pts=[]
    for a in path:
        state=dict(p.neighbors(state))[a];pts.append(state)
    # Try dispersed obstruction candidates; if no replan exists, report unchanged.
    for i in sorted(range(1,len(pts)-1),key=lambda i: abs(i-len(pts)//2))[:30]:
        wall=pts[i]
        if wall in (p.start,p.target):continue
        changed=Maze(p.size,p.open_cells-{wall},p.start,p.target)
        reroute=changed.shortest()
        if reroute.path and changed.apply(reroute.path) and not changed.apply(path):return changed,reroute
    return None,None


def encode_puzzle(p):
    if isinstance(p,ChessEndgame):
        board=['']*64;board[p.wk]='K';board[p.heavy]=p.kind;board[p.bk]='k'
        rows=[]
        for y in range(7,-1,-1):
            row='';blanks=0
            for x in range(8):
                piece=board[y*8+x]
                if piece:
                    if blanks:row+=str(blanks);blanks=0
                    row+=piece
                else:blanks+=1
            if blanks:row+=str(blanks)
            rows.append(row)
        return {'fen':'/'.join(rows)+' w - - 0 1','restricted_kqk_krk':True}
    if isinstance(p,Sudoku):return {'n':p.n,'grid':[list(p.board[i:i+p.n]) for i in range(0,len(p.board),p.n)]}
    if isinstance(p,Maze):return {'n':p.size,'walls':[[r,c] for r in range(p.size) for c in range(p.size) if (r,c) not in p.open_cells],'start':list(p.start),'goal':list(p.target)}
    if isinstance(p,Sliding):return {'n':3,'grid':[list(p.start[i:i+3]) for i in (0,3,6)]}
    if isinstance(p,Sliding4):return {'n':4,'grid':[list(p.start[i:i+4]) for i in (0,4,8,12)]}
    if isinstance(p,Hanoi):return {'discs':p.discs,'state':list(p.start),'goal':list(p.target)}
    raise ValueError(type(p))
