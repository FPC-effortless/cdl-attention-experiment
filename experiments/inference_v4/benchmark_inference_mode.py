"""Separate microbenchmark: cost of Torch inference_mode vs no_grad.
All measured code runs under CPU single-thread; random model; no correctness
checks in the timer. `inference_mode` does not alter model semantics.
"""
from contextlib import nullcontext
from pathlib import Path
import argparse,json,statistics,torch,time,platform
from relational_composition.model import Config
from relational_composition.expressive_model import SerialMultiHopLM, StreamingMultiHop
from relational_composition.hierarchical_model import PagedMultiHopLM, StreamingPagedMultiHop


def run(model,runner_cls,mode,prompt_len,decode_len):
    torch.manual_seed(300)
    x=torch.randint(1,200,(1,prompt_len))
    token=torch.tensor([40])
    context=torch.inference_mode if mode=='inference_mode' else torch.no_grad
    with context():
        runner=runner_cls(model)
        started=time.perf_counter()
        runner.prefill(x)
        prefill=(time.perf_counter()-started)*1000
        started=time.perf_counter()
        for _ in range(decode_len): runner.step(token)
        decode=(time.perf_counter()-started)*1000
    return prefill,decode


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='inference_mode_results.json')
    p.add_argument('--repeats',type=int,default=5)
    args=p.parse_args()
    torch.set_num_threads(1)
    cfg=Config(width=24,layers=2,candidates=2,relations=4,addresses=8)
    result=[]
    for name,model,engine in (
            ('v3',SerialMultiHopLM(cfg).eval(),StreamingMultiHop),
            ('v4',PagedMultiHopLM(cfg,page_bits=20).eval(),StreamingPagedMultiHop)):
        for mode in ('no_grad','inference_mode'):
            run(model,engine,mode,32,8)
            scores=[run(model,engine,mode,512,512) for _ in range(args.repeats)]
            record=dict(architecture=name,mode=mode,prompt=512,output=512,
                prefill_ms=round(statistics.median(t[0] for t in scores),3),
                decode_ms=round(statistics.median(t[1] for t in scores),3))
            result.append(record)
    output=dict(python=platform.python_version(),torch=torch.__version__,
                thread_count=1,median_repeats=args.repeats,
                source='V4 engineering benchmark, untrained',cases=result)
    Path(args.output).write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))

if __name__=='__main__': main()
