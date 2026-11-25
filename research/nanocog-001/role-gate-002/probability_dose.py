"""Pure-Python float64 geometry for an equal-L2 simplex intervention."""
from dataclasses import dataclass
from itertools import permutations
import math

DIM = 4
EPS_DOSE = 1e-12
MIN_FEASIBLE_DOSE = 1e-9
MAX_ABS_COS = .5
BOUNDARY_SAFETY = .99
SUM_ATOL = 2e-14
NORM_RTOL = 3e-7
NORM_ATOL = 4e-16
PERMS = tuple(p for p in permutations(range(DIM)) if p != tuple(range(DIM)))


@dataclass(frozen=True)
class Dose:
    native: tuple[float, ...]
    toward: tuple[float, ...]
    sham: tuple[float, ...]
    requested: float
    radius: float
    native_min: float
    toward_cap: float | None
    sham_cap: float | None
    abs_cosine: float | None
    toward_actual: float
    sham_actual: float
    sham_permutation: tuple[int, ...] | None
    capped: bool
    status: str


def checked(v):
    x = tuple(float(t) for t in v)
    if len(x) != DIM or not all(math.isfinite(t) and 0 <= t <= 1 for t in x):
        raise ValueError("invalid four-way probability vector")
    if abs(math.fsum(x) - 1) > SUM_ATOL:
        raise ValueError("not a normalized probability vector")
    return x


def softmax64(scores):
    x = tuple(float(t) for t in scores)
    if len(x) != DIM or not all(math.isfinite(t) for t in x):
        raise ValueError("invalid gate scores")
    e = tuple(math.exp(t - max(x)) for t in x)
    z = math.fsum(e)
    return checked(t / z for t in e)


def radius_to_boundary(p, u):
    return min(p_i / -u_i for p_i, u_i in zip(p, u, strict=True) if u_i < 0)


def match_probabilities(native, donor):
    p, q = checked(native), checked(donor)
    raw = tuple(b-a for a,b in zip(p,q,strict=True))
    offset = math.fsum(raw)/DIM
    centered = tuple(x-offset for x in raw)
    size = math.hypot(*centered)
    def inert(status, tc=None, sc=None, cos=None, perm=None, cap=False):
        return Dose(p,p,p,size,0.,min(p),tc,sc,cos,0.,0.,perm,cap,status)
    if size <= EPS_DOSE:
        return inert("tiny_native_donor")
    # Retain strict-positive endpoints. Already-unrepresentable softmax mass
    # is a separately counted intent-to-probe no-op, not a hidden exclusion.
    if min(p) == 0.:
        return inert("native_support_zero")
    u = tuple(x/size for x in centered)
    tc = radius_to_boundary(p, u)
    candidates=[]
    for perm in PERMS:
        sham=tuple(u[i] for i in perm)
        cos=min(1.,abs(math.fsum(a*b for a,b in zip(u,sham,strict=True))))
        if cos <= MAX_ABS_COS:
            sc=radius_to_boundary(p, sham)
            r=min(size,BOUNDARY_SAFETY*min(tc,sc))
            candidates.append((-r,cos,perm,sc,sham))
    if not candidates:
        return inert("no_angular_sham",tc)
    neg_r, cos, perm, sc, sham=min(candidates)
    r=-neg_r
    cap = r < size
    if r < MIN_FEASIBLE_DOSE:
        return inert("tiny_common_feasible",tc,sc,cos,perm,cap)
    def point(d):
        out = checked(a+r*b for a,b in zip(p,d,strict=True))
        if min(out) <= 0.:
            raise ArithmeticError("lost strict support")
        actual = math.hypot(*(a-b for a,b in zip(out,p,strict=True)))
        if not math.isclose(actual,r,rel_tol=NORM_RTOL,abs_tol=NORM_ATOL):
            raise ArithmeticError("realized norm differs from intended dose")
        return out,actual
    t,rt = point(u)
    s,rs = point(sham)
    if not math.isclose(rt,rs,rel_tol=NORM_RTOL,abs_tol=NORM_ATOL):
        raise ArithmeticError("realized treatment doses differ")
    return Dose(p,t,s,size,r,min(p),tc,sc,cos,rt,rs,perm,cap,"active")


def match_scores(native, donor):
    return match_probabilities(softmax64(native),softmax64(donor))
