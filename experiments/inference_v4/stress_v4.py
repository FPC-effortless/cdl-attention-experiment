"""V4 long-output smoke + 65K-context correctness/throughput checks.

Includes fixed-token output mode to isolate streaming inference complexity;
this is NOT a trained LM-quality or semantic multi-hop test.
"""
import json, time, gc, platform
from pathlib import Path
import torch
from relational_composition.model import Config
from relational_composition.hierarchical_model import PagedMultiHopLM, StreamingPagedMultiHop


def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='v4_stress.json')
    p.add_argument('--generated',type=int,default=16384)
    p.add_argument('--prompt',type=int,default=65536)
    args=p.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(17)
    config=Config(width=24,layers=2,candidates=2,addresses=8,relations=4)
    model=PagedMultiHopLM(config,page_bits=20).eval()
    results={'environment':dict(python=platform.python_version(),torch=torch.__version__,threads=1),
             'architecture':'V4 page-addressed, untrained, width=24, layers=2, cap=2, page_bits=20'}
    with torch.inference_mode():
        runner=StreamingPagedMultiHop(model)
        t=torch.randint(1,128,(1,128))
        runner.prefill(t)
        fixed=torch.tensor([49])
        checkpoints=[]
        started=time.perf_counter()
        for step in range(1,args.generated+1):
            logits=runner.step(fixed)
            if step in (1,32,128,512,2048,4096,8192,16384,args.generated):
                # differential full-prefix verification outside timing below
                checkpoints.append((step,logits.clone()))
        decode_s=time.perf_counter()-started
        correctness=[]
        for step,actual in checkpoints:
            prefix=torch.cat((t,torch.full((1,step),49,dtype=torch.long)),dim=1)
            full=model(prefix).logits[:,-1]
            diff=(actual-full).abs().max().item()
            correctness.append({'length':step,'max_abs_error':diff,
                                'argmax_equal':bool((actual.argmax(-1)==full.argmax(-1)).all())})
            assert diff < 1e-4
        results['long_output']={'prompt_tokens':128,'generated_tokens':args.generated,
             'decode_seconds':round(decode_s,4),
             'token_steps_per_second':round(args.generated/decode_s,2),
             'checkpoints':correctness,
             'state_capacity':runner.cache.states[0].values.shape[1],
             'index_operations':runner.cache.count.copy()}
        gc.collect()
        prompt=torch.randint(1,128,(1,args.prompt))
        started=time.perf_counter()
        runner.prefill(prompt)
        prefill_s=time.perf_counter()-started
        n=512
        started=time.perf_counter()
        for _ in range(n):
            logits=runner.step(fixed)
        continuation_s=time.perf_counter()-started
        results['long_prompt']={'prompt_tokens':args.prompt,
             'generated_tokens':n,'prefill_seconds':round(prefill_s,4),
             'decode_seconds':round(continuation_s,4),
             'decode_tokens_per_second':round(n/continuation_s,2),
             'state_capacity':runner.cache.states[0].values.shape[1],
             'state_allocated_mib':round(sum(buf.values.numel()*buf.values.element_size()
                             for buf in runner.cache.states)/1048576,3),
             'indexed_writes':runner.cache.count['writes']}
    Path(args.output).write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps({'long_output':results['long_output']['generated_tokens'],
           'tok_s':results['long_output']['token_steps_per_second'],
           'long_prompt':results['long_prompt']['prompt_tokens'],
           'prompt_prefill_s':results['long_prompt']['prefill_seconds']}))

if __name__=='__main__':
    main()
