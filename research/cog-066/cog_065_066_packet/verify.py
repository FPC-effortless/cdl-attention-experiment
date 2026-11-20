"""Independent parser and transition checker; does not import generator or planner."""
import re

RESULT=re.compile(
 r"^(?:parcel (p[0-2]) is (completed|blocked|unknown) at version ([01]) with ([0-3]) of ([23]) steps done"
 r"|at revision ([01]) item (p[0-2]) was (completed|blocked|unknown) with ([0-3]) of ([23]) steps done"
 r"|record for parcel (p[0-2]) reports (completed|blocked|unknown) at version ([01]) and progress ([0-3]) of ([23]))$"
)

def parsed(text):
    m=RESULT.fullmatch(text or "")
    if not m: return None
    f=m.groups()
    if f[0] is not None: row=f[:5]
    elif f[5] is not None: row=(f[6],f[7],f[5],f[8],f[9])
    else: row=f[10:]
    return (int(row[0][1]),row[1],int(row[2]),int(row[3]),int(row[4]))

def observe(world,result):
    n=len(world.bays); st=[0]*n; bay=0; version=0; cost=0; block=world.blocked
    ev=list(result['events'])
    assert len(ev)<=1
    if world.revise_after is not None and len(result['receipts'])>world.revise_after:
        assert ev, "missing observed revision"
    if ev:
        idx,new_v,which=ev[0]
        assert idx==world.revise_after and new_v==1 and which==result['focus']
        assert idx<=len(result['receipts']), "revision before missing receipts"
    for ordinal,(ver, action, state, paid) in enumerate(result['receipts']):
        if ev and ordinal==ev[0][0]:
            version=1;block=ev[0][2]
        assert ver==version, "stale primitive committed"
        if 0<=action<4:
            assert action!=bay and paid==1+abs(bay-action)
            bay=action
        else:
            i=action-4
            assert 0<=i<n and i!=block and st[i]<world.wanted[i]
            assert world.bays[i][st[i]]==bay and paid==1
            st[i]+=1
        assert tuple(st+[bay])==tuple(state)
        cost+=paid
    if ev:version=1;block=ev[0][2]
    expected = 'completed' if tuple(st)==world.wanted else ('unknown' if result['budget'] else 'blocked')
    assert tuple(st+[bay])==tuple(result['state']) and version==result['version'] and cost==result['spent']
    assert result['status']==expected
    if expected=='blocked':
        assert block is not None and st[block]<world.wanted[block]
    i=result['focus']
    return (i,expected,version,st[i],world.wanted[i])

def score(text,truth):
    r=parsed(text)
    return {'strict':r==truth,'facts':sum(a==b for a,b in zip(r,truth)) if r is not None else 0,
            'false_completed': 'completed' in (text or '').split() and truth[1]!='completed',
            'parsed': r}



def exact_cost(world):
    """Independent unpruned Dijkstra oracle for an authorized, static world."""
    import heapq
    toll={(0,0,0,0):0}
    queue=[(0,(0,0,0,0))]
    while queue:
        money,s=heapq.heappop(queue)
        if money!=toll[s]:
            continue
        if tuple(s[:-1])==world.wanted:
            return money
        pos=s[3]
        edges=[(s[:-1]+(v,),1+abs(v-pos)) for v in range(4) if v!=pos]
        for i,progress in enumerate(s[:-1]):
            if progress<world.wanted[i] and world.bays[i][progress]==pos:
                nxt=list(s);nxt[i]+=1
                edges.append((tuple(nxt),1))
        for next_,additional in edges:
            best=money+additional
            if best<toll.get(next_,10**12):
                toll[next_]=best
                heapq.heappush(queue,(best,next_))
    raise AssertionError('authorized static goal unreachable')


def verify_acquisition(data, records):
    wanted={sig:(focus,w) for sig,v,focus,w in data if v=='success'}
    assert len(wanted)==len(records)
    seen=set()
    for row in records:
        sig=tuple(row['source_family']); assert sig not in seen
        seen.add(sig)
        focus,w=wanted[sig]
        r=row['execution']; assert focus==r['focus']
        assert observe(w,r)[1]=='completed'
        assert exact_cost(w)==r['spent']
        assert row['path']==[rec[1] for rec in r['receipts']] and row['verified']
    assert seen==set(wanted)
    return len(seen)
