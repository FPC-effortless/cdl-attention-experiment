"""Native CPU reductions and batched exact scoped attention over the same relevant values."""
import json
from pathlib import Path
from time import perf_counter_ns
import resource
import numpy as np
import torch
from torch.nn.functional import scaled_dot_product_attention

rows=[]
for size in (128,2048,32768,131072,1048576):
    rng=np.random.default_rng(59001+size)
    values=torch.from_numpy(rng.integers(0,2,size,dtype=np.int32)).float().view(32,size//32)
    offsets=torch.arange(32)*values.shape[1]
    for batch in (1,32,128):
        keys=torch.tensor(rng.integers(32,size=batch))
        oracle=values[keys].double().mean(1)
        for threads in (1,2):
            torch.set_num_threads(threads)
            for arm in ('gather_mean','sdpa'):
                times=[]
                for _ in range(8):
                    t=perf_counter_ns()
                    scoped=values.index_select(0,keys)
                    if arm=='sdpa':
                        v=scoped[:,None,:,None]
                        result=scaled_dot_product_attention(torch.zeros(batch,1,1,1),torch.zeros_like(v),v).flatten()
                    else:result=scoped.mean(1)
                    times.append(perf_counter_ns()-t)
                    assert (result.double()-oracle).abs().max()<1e-5
                rows.append(dict(size=size,batch=batch,cpu_threads=threads,arm=arm,median_ns=float(np.median(times[1:]))))
out=dict(rows=rows,process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
         cuda=torch.cuda.is_available(),device_count=torch.cuda.device_count())
(Path(__file__).resolve().parent/'results/parallel-cpu.json').write_text(json.dumps(out))
