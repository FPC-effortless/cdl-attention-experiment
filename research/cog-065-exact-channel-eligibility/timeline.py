"""Authorize an observation snapshot before touching receipt payload semantics."""
from dataclasses import dataclass
@dataclass(frozen=True)
class Envelope:
    slot: str
    tick: int
    sequence: int
    payload: object

def snapshot(envelopes, cutoff, decode):
    if type(cutoff) is not int or cutoff < 0:
        raise ValueError("cutoff")
    occupants={}; seen={}
    for e in envelopes:
        if type(e.tick) is not int or e.tick<0:
            raise ValueError("bad envelope availability")
        if e.tick>cutoff:
            continue
        if type(e.slot) is not str or not e.slot or type(e.sequence) is not int or e.sequence<0:
            raise ValueError("bad admitted route")
        loc=(e.slot,e.tick,e.sequence)
        if loc in seen and seen[loc] is not e:
            # No comparison of superseded or malformed real payloads. Duplicate
            # delivery must reuse the same immutable admitted envelope identity.
            raise ValueError("ambiguous authorized packet position")
        seen[loc]=e
        old=occupants.get(e.slot)
        if old is None or (e.tick,e.sequence)>(old.tick,old.sequence):
            occupants[e.slot]=e
    return {key:decode(key,e.payload,e.tick) for key,e in sorted(occupants.items())}
