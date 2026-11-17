"""021 prefix-only inputs; annotations are labels, never conditioner features."""
import hashlib,json,re
from pathlib import Path
import torch
from surface_patterns import PATTERNS
CAPACITY=16
CONDITION_DIM=392

def turns(record):
    previous=0
    for index,answer in enumerate(record['answers']):
        start,end=answer['start'],answer['end']
        if not previous<=start<=end<=len(record['text']):raise ValueError('answer offsets')
        # Target content and hidden metadata do not determine inference input.
        chunk=record['text'][previous:start]
        if previous:
            if not chunk.startswith('\n'):raise ValueError('answer separator contract')
            chunk=chunk[1:] # separator is fed with previous attempted output exactly once
        yield index,chunk.encode('utf8'),answer
        previous=end

def padded(feature):
    if feature.ndim!=2 or feature.shape[1]>CONDITION_DIM:raise ValueError('feature width')
    return torch.nn.functional.pad(feature,(0,CONDITION_DIM-feature.shape[1]))

class ExactState:
    """Supplied grammar interpreter, NOT a learned parser/CSM.
    Maps identities by first appearance, including names not seen during training.
    Consumes only external instructions, never generated answers or target metadata.
    """
    def __init__(self):self.names=[];self.records={};self.query=None
    def identity(self,name):
        if name not in self.names:
            if len(self.names)>=CAPACITY:raise ValueError('capacity exceeded')
            self.names.append(name)
        return self.names.index(name)
    def read(self,i):
        seen=set()
        while i in self.records:
            if i in seen:return None
            seen.add(i);kind,value=self.records[i]
            if kind=='value':return value
            i=value
        return None
    def consume(self,external):
        for line in external.decode('utf8').splitlines():
            line=line.strip();found=[]
            for op,patterns in PATTERNS.items():
                for pattern in patterns:
                    if m:=re.fullmatch(pattern,line):found.append((op,m.groupdict()))
            if not found:continue # declared filler and Answer prefix
            if len(found)!=1:raise ValueError('ambiguous grammar')
            op,args=found[0]
            # Canonical identity order follows textual occurrence, not role priority.
            names=[v for k,v in args.items() if k in ('dst','src')]
            for name in sorted(names,key=line.index):self.identity(name)
            d=self.identity(args['dst']);s=self.identity(args['src']) if 'src' in args else None
            if op=='set':self.records[d]=('value',args['value'].encode('ascii'))
            elif op=='copy':
                value=self.read(s)
                if value is None:self.records.pop(d,None)
                else:self.records[d]=('value',value)
            elif op=='link':self.records[d]=('ref',s)
            elif op=='delete':self.records.pop(d,None)
            elif op=='query':self.query=d
    def feature(self,device='cpu'):
        x=torch.zeros(1,CONDITION_DIM,device=device);stride=3+3+CAPACITY
        for i in range(CAPACITY):
            entry=self.records.get(i);offset=i*stride
            x[0,offset+(0 if entry is None else 1 if entry[0]=='value' else 2)]=1
            if entry is None:continue
            kind,value=entry
            if kind=='value':
                if len(value)!=3:raise ValueError('three-byte payload contract')
                for j,b in enumerate(value):x[0,offset+3+j]=b/255.
            else:x[0,offset+6+value]=1
        if self.query is not None:x[0,CAPACITY*stride+self.query]=1
        return x

def input_hash(record):
    return hashlib.sha256(b''.join(chunk for _,chunk,_ in turns(record))).hexdigest()

def read_train(path):
    rows=[json.loads(x) for x in Path(path).read_text().splitlines()]
    if not rows or any(not r['id'].startswith('train-') for r in rows):raise ValueError('TRAIN-only cache required')
    return rows

def render_target(answer):
    value=answer['target']
    if value=='unknown':return 'No code is currently assigned.'
    if not re.fullmatch(r'[A-Z2-9]{3}',value):raise ValueError('payload contract')
    return f'The current code is {value}.'

def extract_payload(text):
    if text=='No code is currently assigned.':return 'unknown'
    matches=re.findall(r'(?<![A-Za-z0-9])([A-Z2-9]{3})(?![A-Za-z0-9])',text)
    return matches[0] if len(matches)==1 else None
