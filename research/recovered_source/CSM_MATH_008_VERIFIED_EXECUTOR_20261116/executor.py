"""Typed reference executor. Exact immutable payloads; no neural policy claimed."""
from dataclasses import dataclass
@dataclass(frozen=True)
class Cell:
    kind: str = 'absent'
    payload: object = None

class Memory:
    def __init__(self,n):
        if n < 1: raise ValueError('positive capacity required')
        self.cells=[Cell() for _ in range(n)]
    def check(self,i):
        if not isinstance(i,int) or not 0<=i<len(self.cells): raise IndexError(i)
    def read(self,i):
        self.check(i)
        for _ in range(len(self.cells)):
            c=self.cells[i]
            if c.kind=='absent': return None
            if c.kind=='value': return c.payload
            i=c.payload
        return None
    def apply(self,op,i,j=None,v=None):
        self.check(i)
        if op in ('COPY','LINK'): self.check(j)
        if op=='QUERY': return self.read(i)
        if op=='SET':
            if type(v) is not int or v < 0: raise ValueError('reference payload must be an immutable natural number')
            self.cells[i]=Cell('value',v)
        elif op=='COPY':
            value=self.read(j) # before destination mutation, including self-copy
            self.cells[i]=Cell() if value is None else Cell('value',value)
        elif op=='LINK': self.cells[i]=Cell('ref',j)
        elif op=='DELETE': self.cells[i]=Cell()
        else: raise ValueError(op)
