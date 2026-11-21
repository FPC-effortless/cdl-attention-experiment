"""Time a fresh Python process from launch through detokenization and exit."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def worker(a):
    import torch
    from tokenizers import Tokenizer
    from runtime import Runtime
    torch.set_num_threads(2)
    tok=Tokenizer.from_file(str(a.tokenizer))
    m=Runtime.load(a.base,a.adapter,a.disabled)
    encoded=tok.encode(sys.stdin.read()).ids
    pred,state=m.advance(torch.tensor([encoded]),last_only=True)
    ids=[]
    for _ in range(32):
        token=pred[:,-1].argmax(-1,keepdim=True)
        ids.append(token.item())
        pred,state=m.advance(token,state,last_only=True)
    print(json.dumps(dict(input_tokens=len(encoded),output_tokens=ids,text=tok.decode(ids))))


def measure(a):
    from tokenizers import Tokenizer
    import numpy as np
    tok=Tokenizer.from_file(str(a.tokenizer))
    arr=np.load(a.data/'test.npy',mmap_mode='r')
    offs=np.load(a.data/'test.offsets.npy')
    lo=next(int(lo)+32 for lo,hi in zip(offs,offs[1:]) if hi-lo>17000)
    prompt=tok.decode(arr[lo:lo+1024].tolist())
    args=[sys.executable,__file__,'--worker','--base',str(a.base),'--adapter',str(a.adapter),
          '--tokenizer',str(a.tokenizer)]
    if a.disabled:
        args.append('--disabled')
    records=[]
    for _ in range(3):
        start=time.perf_counter()
        proc=subprocess.run(args,input=prompt,text=True,capture_output=True,check=True)
        records.append(dict(wall_s=time.perf_counter()-start,**json.loads(proc.stdout)))
    name=a.adapter.stem+('_off' if a.disabled else '')
    (a.out/(name+'.process.json')).write_text(json.dumps(records)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for k in ('base','adapter','tokenizer'):
        parser.add_argument('--'+k,type=Path,required=True)
    parser.add_argument('--data',type=Path)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--disabled',action='store_true')
    parser.add_argument('--worker',action='store_true')
    a=parser.parse_args()
    if a.worker:
        worker(a)
    else:
        measure(a)
