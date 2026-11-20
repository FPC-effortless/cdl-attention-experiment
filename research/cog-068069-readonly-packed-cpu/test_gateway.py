from gateway import Authority,admissible,select,serialize
h={k:[(3,[1.]),(4,[2.]),(5,[3.])] for k in ('a','b')}
a=Authority('a','s',3,1,(1,True));b=Authority('b','s',2,0,(1,True))
revoke=Authority('a','s',5,2,(1,False))
bad_same_tick=Authority('a','s',5,1,object())
future=Authority('a','s',6,99,object());other=Authority('a','other',5,100,object())
assert admissible([a,revoke,b,bad_same_tick,future,other,b],'s',5,h)=={'b':((1.,),(2.,),(3.,))}
assert admissible([future,b,bad_same_tick,a,other,revoke],'s',5,h)=={'b':((1.,),(2.,),(3.,))}
assert select({'b':.2},{'b':h['b']})=='b'
assert select({'b':.2,'a':.2},h)=='a'
assert serialize('b',{'b':h['b']})=={'kind':'prediction_only_not_observed','lane':'b'}
assert serialize(None,{})=={'kind':'prediction_only_not_observed','lane':None}
for apply in [lambda: admissible([b,Authority('b','s',2,0,(1,False))],'s',5,h),
              lambda: serialize('a',{'b':h['b']}),
              lambda: select({'b':float('nan')},{'b':h['b']}),
              lambda: select({'a':.8},{'b':h['b']})]:
    try:apply();raise AssertionError('invalid path accepted')
    except ValueError:pass
print('PASS constructed gateway shape fixtures: same-tick revocation, hidden future, scope partition, stable ties, no status claims; not a runtime benchmark')
