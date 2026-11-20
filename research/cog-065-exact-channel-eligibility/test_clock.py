from timeline import Envelope, snapshot
from rules import receive,reference,ROLES,KNOWN
from codec import encode,parse
a=Envelope('state',5,1,object())
b=Envelope('state',5,2,'ready')
f=Envelope('state',6,0,object())
calls=[]
def decode(s,p,tick):
    calls.append((s,p))
    if type(p) is not str: raise ValueError('bad payload')
    return p
assert snapshot([a,b,f,b],5,decode)=={'state':'ready'}
assert calls==[('state','ready')]
assert snapshot([f,a,b],5,decode)=={'state':'ready'}
assert snapshot([f,a,b],4,decode)=={}
calls.clear()
try:snapshot([b,Envelope('state',5,2,'different')],5,decode); raise AssertionError('ambiguous order accepted')
except ValueError:pass
assert calls==[]
try:snapshot([b,f],6,decode); raise AssertionError('bad occupied payload accepted')
except ValueError:pass
x={'status':'completed','proof':'accepted','authority':'current','source':'verified','scope':'matched','freshness':'current','progress':'started'}
for role in ROLES:
    assert receive(role,{},parse(encode(x,x,64),64))==reference(role,x)
    assert receive(role,{}, {'status':'completed'})=='seek evidence'
    assert receive(role,{'scope':'mismatch'},parse(encode(x,x,64),64))=='seek evidence'
print('PASS occupation/availability before parse, order independence, closed same-tick, conservative receiver fixtures; no held cases')
