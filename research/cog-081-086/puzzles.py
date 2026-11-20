"""COG-ATTN-081 puzzle environments and reference controls.

These reference solvers are NOT the COG-ATTN model. They create deterministic,
fully inspectable benchmarks and establish achievable verified baselines.
"""
from __future__ import annotations
from dataclasses import dataclass
import heapq
import random
from collections import deque
from time import perf_counter_ns

@dataclass
class Solve:
    path: list
    expanded: int
    elapsed_ns: int


def bfs(start, goal, next_states):
    begin = perf_counter_ns()
    q = deque([start]); parent={start:None}; actions={}; expanded=0
    while q:
        s=q.popleft(); expanded+=1
        if goal(s):
            path=[]
            while parent[s] is not None:
                path.append(actions[s]);s=parent[s]
            return Solve(list(reversed(path)),expanded,perf_counter_ns()-begin)
        for action,nxt in next_states(s):
            if nxt not in parent:
                parent[nxt]=s; actions[nxt]=action;q.append(nxt)
    return Solve([],expanded,perf_counter_ns()-begin)


def astar(start,goal,next_states,heuristic):
    begin=perf_counter_ns(); counter=0; heap=[(heuristic(start),0,counter,start)]; dist={start:0}; parent={start:None}; acts={};expanded=0
    while heap:
        _,g,_,s=heapq.heappop(heap)
        if dist[s]!=g:continue
        expanded+=1
        if goal(s):
            path=[]
            while parent[s] is not None:
                path.append(acts[s]);s=parent[s]
            return Solve(list(reversed(path)),expanded,perf_counter_ns()-begin)
        for a,n in next_states(s):
            ng=g+1
            if ng<dist.get(n,10**20):
                dist[n]=ng;parent[n]=s;acts[n]=a;counter+=1
                heapq.heappush(heap,(ng+heuristic(n),ng,counter,n))
    return Solve([],expanded,perf_counter_ns()-begin)

@dataclass(frozen=True)
class Maze:
    size:int
    open_cells:frozenset
    start:tuple=(1,1)
    target:tuple|None=None
    def __post_init__(self):
        if self.target is None:object.__setattr__(self,'target',(self.size-2,self.size-2))
    def neighbors(self,p):
        for a,(dr,dc) in [('U',(-1,0)),('D',(1,0)),('L',(0,-1)),('R',(0,1))]:
            n=(p[0]+dr,p[1]+dc)
            if n in self.open_cells:yield a,n
    def apply(self,path):
        s=self.start
        for a in path:
            nxt=dict(self.neighbors(s))
            if a not in nxt:return False
            s=nxt[a]
        return s==self.target
    def shortest(self):return bfs(self.start,lambda s:s==self.target,self.neighbors)
    def greed(self,limit=200):
        s=self.start; seen={s};path=[]
        for _ in range(limit):
            if s==self.target:break
            options=[(abs(n[0]-self.target[0])+abs(n[1]-self.target[1]),a,n) for a,n in self.neighbors(s) if n not in seen]
            if not options:return False
            _,a,s=min(options);seen.add(s);path.append(a)
        return self.apply(path)

def make_maze(seed,size,extra=0.19):
    """DFS corridors plus intentional loops; deterministic from seed."""
    r=random.Random(seed);open_cells={(1,1)};stack=[(1,1)]
    while stack:
        y,x=stack[-1]
        opts=[]
        for dy,dx in [(-2,0),(2,0),(0,-2),(0,2)]:
            ny,nx=y+dy,x+dx
            if 0<ny<size-1 and 0<nx<size-1 and (ny,nx) not in open_cells:opts.append((ny,nx,(y+dy//2,x+dx//2)))
        if opts:
            ny,nx,mid=r.choice(opts);open_cells.add(mid);open_cells.add((ny,nx));stack.append((ny,nx))
        else:stack.pop()
    for y in range(1,size-1):
        for x in range(1,size-1):
            if (y,x) not in open_cells and r.random()<extra:
                open_cells.add((y,x))
    return Maze(size,frozenset(open_cells))

def shift_maze(maze):
    path=maze.shortest().path
    if len(path)<=2:return None
    state=maze.start; visited=[]
    for a in path:
        state=dict(maze.neighbors(state))[a];visited.append(state)
    for cell in visited[:-1]:
        opens=set(maze.open_cells);opens.remove(cell)
        changed=Maze(maze.size,frozenset(opens),maze.start,maze.target)
        if changed.shortest().path and not changed.apply(path):return changed
    return None

@dataclass(frozen=True)
class Hanoi:
    discs:int
    @property
    def start(self):return tuple(0 for _ in range(self.discs))
    @property
    def target(self):return tuple(2 for _ in range(self.discs))
    def neighbors(self,s):
        tops={}
        for d,p in enumerate(s):tops.setdefault(p,d)  # disc 0 is smallest
        for p,d in tops.items():
            for dest in range(3):
                if p!=dest and (dest not in tops or d<tops[dest]):
                    state=list(s);state[d]=dest
                    yield (p,dest),tuple(state)
    def shortest(self):return bfs(self.start,lambda s:s==self.target,self.neighbors)
    def apply(self,path):
        s=self.start
        for a in path:
            choices=dict(self.neighbors(s))
            if a not in choices:return False
            s=choices[a]
        return s==self.target
    def greed(self):
        s=self.start;visited={s}
        for _ in range(3**self.discs):
            if s==self.target:return True
            opts=[(a,t) for a,t in self.neighbors(s) if t not in visited]
            if not opts:return False
            _,s=opts[0];visited.add(s)
        return False

@dataclass(frozen=True)
class Sliding:
    start:tuple
    target=tuple(range(1,9))+(0,)
    def neighbors(self,s):
        z=s.index(0); r,c=divmod(z,3)
        for a,(dr,dc) in [('U',(-1,0)),('D',(1,0)),('L',(0,-1)),('R',(0,1))]:
            rr,cc=r+dr,c+dc
            if 0<=rr<3 and 0<=cc<3:
                j=rr*3+cc;v=list(s);v[z],v[j]=v[j],v[z]
                yield a,tuple(v)
    @staticmethod
    def heuristic(s):
        return sum(abs(i//3-(v-1)//3)+abs(i%3-(v-1)%3) for i,v in enumerate(s) if v)
    def shortest(self):return astar(self.start,lambda s:s==self.target,self.neighbors,self.heuristic)
    def apply(self,path):
        s=self.start
        for a in path:
            choices=dict(self.neighbors(s))
            if a not in choices:return False
            s=choices[a]
        return s==self.target
    def greed(self,limit=60):
        s=self.start;seen={s}
        for _ in range(limit):
            if s==self.target:return True
            opts=[(self.heuristic(t),a,t) for a,t in self.neighbors(s) if t not in seen]
            if not opts:return False
            _,_,s=min(opts);seen.add(s)
        return s==self.target

def make_sliding(seed,steps):
    r=random.Random(seed);p=Sliding(Sliding.target);s=p.start;prev=None
    for _ in range(steps):
        opts=[t for _,t in p.neighbors(s) if t!=prev]
        if not opts:opts=[t for _,t in p.neighbors(s)]
        prev,s=s,r.choice(opts)
    return Sliding(s)

@dataclass(frozen=True)
class Sudoku:
    n:int
    board:tuple
    def __post_init__(self):
        assert len(self.board)==self.n*self.n
    def choices(self,s,i):
        if s[i]:return ()
        n=self.n;k=int(n**0.5);row,col=divmod(i,n)
        used=set(s[row*n:row*n+n]);used.update(s[col::n]);
        used.update(s[y*n+x] for y in range(row//k*k,row//k*k+k) for x in range(col//k*k,col//k*k+k))
        return tuple(v for v in range(1,n+1) if v not in used)
    def solve(self,max_solutions=1):
        started=perf_counter_ns();expanded=0;solutions=[]
        def dfs(s,path):
            nonlocal expanded
            if len(solutions)>=max_solutions:return
            expanded+=1
            holes=[(self.choices(s,i),i) for i,v in enumerate(s) if v==0]
            if not holes:solutions.append(path);return
            choices,i=min(holes,key=lambda p:len(p[0]))
            for value in choices:
                nxt=list(s);nxt[i]=value;dfs(tuple(nxt),path+[(i,value)])
                if len(solutions)>=max_solutions:return
        dfs(self.board,[])
        return Solve(solutions[0] if solutions else [],expanded,perf_counter_ns()-started),len(solutions)
    def apply(self,path):
        s=list(self.board)
        for i,value in path:
            if i<0 or i>=len(s) or s[i] or value not in self.choices(tuple(s),i):return False
            s[i]=value
        return all(s) and all(not self.choices(tuple(s),i) for i in range(len(s))) and self.valid(tuple(s))
    def valid(self,s):
        n=self.n;k=int(n**0.5)
        if any(v not in range(1,n+1) for v in s):return False
        goal=list(range(1,n+1))
        for r in range(n):
            if sorted(s[r*n:r*n+n])!=goal or sorted(s[r::n])!=goal:return False
        for y in range(0,n,k):
            for x in range(0,n,k):
                if sorted(s[(y+dy)*n+x+dx] for dy in range(k) for dx in range(k))!=goal:return False
        return True
    def greed(self):
        s=list(self.board)
        while 0 in s:
            singles=[(i,options[0]) for i,v in enumerate(s) if v==0 if len(options:=self.choices(tuple(s),i))==1]
            if not singles: return False
            for i,v in singles:s[i]=v
        return self.valid(tuple(s))

def make_sudoku(seed,n,remove):
    r=random.Random(seed);k=int(n**0.5)
    perm=list(range(1,n+1));r.shuffle(perm)
    rows=[(g*k+sub) for g in r.sample(range(k),k) for sub in r.sample(range(k),k)]
    cols=[(g*k+sub) for g in r.sample(range(k),k) for sub in r.sample(range(k),k)]
    solution=[perm[(y*k+y//k+x)%n] for y in rows for x in cols]
    board=list(solution);order=list(range(n*n));r.shuffle(order)
    for i in order:
        if board.count(0)>=remove:break
        prior=board[i];board[i]=0
        _,solutions=Sudoku(n,tuple(board)).solve(2)
        if solutions!=1:board[i]=prior
    return Sudoku(n,tuple(board))

# Restricted legal chess endgames: K+Q vs K and K+R vs K, white to move.
# This is a declared subset, not a complete chess engine.
def adjacent(a,b):return max(abs(a//8-b//8),abs(a%8-b%8))==1

def aligned(a,b,kind):
    dr,dc=abs(a//8-b//8),abs(a%8-b%8)
    return dr==0 or dc==0 or (kind=='Q' and dr==dc)

def line_attack(a,b,wk,kind):
    if a<0 or not aligned(a,b,kind):return False
    dr=(b//8>a//8)-(b//8<a//8);dc=(b%8>a%8)-(b%8<a%8)
    r,c=a//8+dr,a%8+dc
    while (r,c)!=(b//8,b%8):
        if 8*r+c==wk:return False
        r+=dr;c+=dc
    return True

@dataclass(frozen=True)
class ChessEndgame:
    wk:int
    heavy:int
    bk:int
    kind:str='Q'
    def white_moves(self):
        options=[]
        for dr in [-1,0,1]:
            for dc in [-1,0,1]:
                if not (dr or dc):continue
                r,c=divmod(self.wk,8);r+=dr;c+=dc
                if not 0<=r<8 or not 0<=c<8:continue
                t=r*8+c
                if t!=self.heavy and t!=self.bk and not adjacent(t,self.bk):options.append(('K',self.wk,t,ChessEndgame(t,self.heavy,self.bk,self.kind)))
        ds=[(-1,0),(1,0),(0,-1),(0,1)]+([(-1,-1),(-1,1),(1,-1),(1,1)] if self.kind=='Q' else [])
        for dr,dc in ds:
            r,c=divmod(self.heavy,8);r+=dr;c+=dc
            while 0<=r<8 and 0<=c<8:
                t=r*8+c
                if t==self.wk or t==self.bk:break
                options.append((self.kind,self.heavy,t,ChessEndgame(self.wk,t,self.bk,self.kind)))
                r+=dr;c+=dc
        return options
    def black_checked(self):return adjacent(self.wk,self.bk) or line_attack(self.heavy,self.bk,self.wk,self.kind)
    def black_replies(self):
        moves=[];r0,c0=divmod(self.bk,8)
        for dr in [-1,0,1]:
            for dc in [-1,0,1]:
                if not (dr or dc):continue
                r,c=r0+dr,c0+dc
                if not(0<=r<8 and 0<=c<8):continue
                dest=r*8+c
                if dest==self.wk or adjacent(dest,self.wk):continue
                if dest!=self.heavy and line_attack(self.heavy,dest,self.wk,self.kind):continue
                moves.append(dest)
        return moves
    def mate(self):return self.black_checked() and not self.black_replies()
    def mates(self):return [(name,old,new) for name,old,new,child in self.white_moves() if child.mate()]
    def apply(self,path):
        if len(path)!=1:return False
        return any((name,old,new)==path[0] and child.mate() for name,old,new,child in self.white_moves())
    def shortest(self):
        before=perf_counter_ns();moves=self.white_moves();mates=[(k,a,b) for k,a,b,c in moves if c.mate()]
        return Solve(mates[:1],len(moves),perf_counter_ns()-before)
    def greed(self):
        moves=self.white_moves();return bool(moves) and moves[0][3].mate()

def chess_candidates(kind):
    # Reuse a deterministic enumerated set, sampled independently by RNG for each trial.
    pool=[]
    for bk in range(64):
        if bk//8 not in (0,7) and bk%8 not in (0,7):continue
        for wk in range(64):
            if wk==bk or adjacent(wk,bk):continue
            for heavy in range(64):
                if heavy in (wk,bk):continue
                pos=ChessEndgame(wk,heavy,bk,kind)
                if pos.black_checked():continue
                if len(pos.mates())==1:pool.append(pos)
    return pool
