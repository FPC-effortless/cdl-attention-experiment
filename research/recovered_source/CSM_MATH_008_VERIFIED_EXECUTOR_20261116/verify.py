import itertools,json,hashlib,subprocess,time, pathlib
from executor import Memory,Cell
P=pathlib.Path(__file__).resolve().parent
LEAN='/tmp/lean-4.15.0-linux/bin/lean'
def oracle(cells,i):
    seen=set()
    while i not in seen:
        seen.add(i); c=cells[i]
        if c.kind=='absent':return None
        if c.kind=='value':return c.payload
        i=c.payload
    return None

def run():
 start=time.time(); counts=dict(states=0,queries=0,transitions=0,invalid_rejections=0,mutation_witnesses=0)
 cells=[Cell(),Cell('value',0),Cell('value',1)]+[Cell('ref',i) for i in range(3)]
 rows=[]
 for state in itertools.product(cells,repeat=3):
  counts['states']+=1;m=Memory(3);m.cells=list(state)
  for i in range(3):
   counts['queries']+=1;assert m.read(i)==oracle(state,i)
   before=m.cells.copy();assert m.apply('QUERY',i)==oracle(state,i);assert m.cells==before
  for op,i,j,v in ([('SET',i,None,v) for i in range(3) for v in (0,1)]+[('DELETE',i,None,None) for i in range(3)]+[(op,i,j,None) for op in ('COPY','LINK') for i in range(3) for j in range(3)]):
   m.cells=list(state);m.apply(op,i,j,v); counts['transitions']+=1
   expected=list(state)
   if op=='SET':expected[i]=Cell('value',v)
   elif op=='DELETE':expected[i]=Cell()
   elif op=='LINK':expected[i]=Cell('ref',j)
   else:
    val=oracle(state,j);expected[i]=Cell() if val is None else Cell('value',val)
   assert m.cells==expected
   assert all(m.read(k)==oracle(expected,k) for k in range(3))
  def expr(c):return 'Cell.absent' if c.kind=='absent' else f'Cell.{"value" if c.kind=="value" else "ref"} {c.payload}'
  e='fun i => '+''.join(f'if i = {i} then {expr(c)} else ' for i,c in enumerate(state))+'Cell.absent'
  for i in range(3):
   val=oracle(state,i);out='none' if val is None else f'some {val}'
   rows.append(f'example : TypedMemory.read 3 ({e} : TypedMemory.Memory Nat) {i} = {out} := by decide')
 m=Memory(3)
 for bad in [-1,3]:
  for op in ['SET','COPY','LINK','DELETE','QUERY']:
   before=m.cells.copy()
   try:m.apply(op,bad,0,1);raise AssertionError('accepted invalid')
   except IndexError:pass
   assert before==m.cells;counts['invalid_rejections']+=1
 for bad in [-1,3]:
  for op in ['COPY','LINK']:
   before=m.cells.copy()
   try:m.apply(op,0,bad);raise AssertionError('accepted invalid source')
   except IndexError:pass
   assert before==m.cells;counts['invalid_rejections']+=1
 for bad in [None,[],{},[1],-1,1.5,True]:
  before=m.cells.copy()
  try:m.apply('SET',0,v=bad);raise AssertionError('accepted invalid payload')
  except ValueError:pass
  assert before==m.cells;counts['invalid_rejections']+=1
 # real defective operations, not just comparisons of prewritten constants
 def bad_copy(m,i,j): m.apply('LINK',i,j)
 m.apply('SET',0,v=0);bad_copy(m,1,0);m.apply('SET',0,v=1)
 assert m.read(1)!=0;counts['mutation_witnesses']+=1
 def bad_delete(m,i): pass
 bad_delete(m,0);assert m.read(0) is not None;counts['mutation_witnesses']+=1
 def bad_query(m,i):m.apply('DELETE',i);return None
 before=m.cells.copy();bad_query(m,0);assert m.cells!=before;counts['mutation_witnesses']+=1
 parity=P/'Parity.lean';parity.write_text('import TypedMemory\nopen TypedMemory\n'+'\n'.join(rows)+'\n')
 env=__import__('os').environ.copy();env['LEAN_PATH']=str(P)
 runs=[]
 for args in [[LEAN,'-o',str(P/'TypedMemory.olean'),str(P/'TypedMemory.lean')],[LEAN,str(parity)]]:
  r=subprocess.run(args,capture_output=True,text=True,env=env);runs.append(dict(command=args,exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr));assert r.returncode==0
 # intentionally false theorem must be rejected by kernel elaboration, not assumed
 mutation=P/'Mutation.lean';mutation.write_text('import TypedMemory\nopen TypedMemory\ndef badCopy (m : Memory Nat) (i j : Nat) := link m i j\nexample : read 3 (set (badCopy (set empty 0 7) 1 0) 0 9) 1 = some 7 := by decide\n')
 r=subprocess.run([LEAN,str(mutation)],capture_output=True,text=True,env=env);assert r.returncode!=0
 runs.append(dict(command=[LEAN,str(mutation)],expected_failure=True,exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr))
 result=dict(counts=counts,lean_concrete_examples=len(rows),runs=runs,elapsed=time.time()-start,scope='executor only; no neural models/training; exhaustive N=3 values0,1')
 (P/'RESULTS.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='runs'}))
if __name__=='__main__':run()
