"""Local synthetic parcel workflow. No network or physical side effects."""
from dataclasses import dataclass

@dataclass(frozen=True)
class World:
    bays: tuple[tuple[int, ...], ...]
    wanted: tuple[int, ...]
    names: tuple[str, ...]
    blocked: int | None = None
    revise_after: int | None = None

def initial(w):
    return (0,) * len(w.bays) + (0,)

def done(w,s):
    return all(a >= b for a,b in zip(s[:-1],w.wanted))

def step(w,s,a,blocked=None):
    k = len(w.bays)
    if not 0 <= a < 4+k:
        return None
    if a < 4:
        if a == s[-1]:
            return None
        return s[:-1]+(a,), 1+abs(s[-1]-a)
    i=a-4
    if i==blocked or s[i]>=w.wanted[i] or w.bays[i][s[i]]!=s[-1]:
        return None
    return s[:i]+(s[i]+1,)+s[i+1:],1

def h(w,s):
    required=[bay for i,n in enumerate(w.wanted) for bay in w.bays[i][s[i]:n]]
    if not required:
        return 0
    lo,hi=min(required),max(required)
    return len(required)+hi-lo+min(abs(s[-1]-lo),abs(s[-1]-hi))

