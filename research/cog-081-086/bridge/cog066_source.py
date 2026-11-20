"""COG-066 skill acquisition function, vendored verbatim from COG-ATTN-071–080.

Upstream: research/cog-071-080/vendor/packet/planning.py
Revision: 975732f1081d4e5c98c460df1cfa067c48af213d
Only the acquire() function is imported here. Planning dynamics are separately
adapted to each puzzle; those adapters are new code, not original COG-ATTN.
"""
from collections import Counter, defaultdict

def acquire(trace_rows,limit=16):
    """Compression proxy: frequency times saved choices minus dictionary cost.
    Only primitive successful traces and source-family identity are used.
    """
    counts=Counter()
    groups=defaultdict(set)
    for row in trace_rows:
        assert row['verified']
        path=tuple(row['path'])
        for n in (2,3,4):
            for i in range(len(path)-n+1):
                seq=path[i:i+n]
                counts[seq]+=1
                groups[seq].add(row['source_family'])
    eligible=[seq for seq in counts if len(groups[seq])>=3 and counts[seq]>=8]
    eligible.sort(key=lambda seq:(-((len(seq)-1)*counts[seq]-len(seq)),seq))
    return tuple(eligible[:limit])
