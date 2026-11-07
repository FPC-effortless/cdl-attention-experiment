"""Reproducible CPU performance diagnostic for LRC-V3/V4 and graph compilation.

All timings exclude correctness checks. Models are random, untrained and are
benchmarked on FIXED non-PAD tokens so routing variability does not change the
input. These benchmarks are not comparisons against production LM systems.
"""
import argparse, json, math, platform, statistics, time
from pathlib import Path
import numpy as np
import torch
from relational_composition.model import Config
from relational_composition.expressive_model import SerialMultiHopLM, StreamingMultiHop
from relational_composition.hierarchical_model import PagedMultiHopLM, StreamingPagedMultiHop
from relational_composition.hierarchical_index import PagedCausalIndex
from relational_composition.program_runtime import VersionedProgramEngine, serial_batch


def observe_once(model, cls, prompt, output, batch):
    torch.manual_seed(194)
    x=torch.randint(1,128,(batch,prompt))
    next_token=torch.full((batch,),49,dtype=torch.long)
    runner=cls(model)
    started=time.perf_counter()
    runner.prefill(x)
    prefill=(time.perf_counter()-started)*1000
    started=time.perf_counter()
    for _ in range(output):
        runner.step(next_token)
    decode=(time.perf_counter()-started)*1000
    return dict(prefill_ms=prefill, decode_ms=decode,
                decode_tokens_per_second=1000*batch*output/decode,
                total_ms=prefill+decode, cache_records=runner.cache.position,
                operations=runner.cache.count.copy())


def bench_models(repeats):
    cases=[]
    specs=[(128,32,1),(128,256,1),(128,1024,1),(512,128,1),
           (512,512,1),(256,256,4)]
    for name,typ,engine in (("v3",SerialMultiHopLM,StreamingMultiHop),
                            ("v4",PagedMultiHopLM,StreamingPagedMultiHop)):
        torch.manual_seed(13)
        cfg=Config(width=24,layers=2,candidates=2,relations=4,addresses=8)
        model=(typ(cfg,page_bits=20) if name=='v4' else typ(cfg)).eval()
        parameters=sum(x.numel() for x in model.parameters())
        for prompt,out,batch in specs:
            observe_once(model,engine,8,3,batch)  # warmup
            trials=[observe_once(model,engine,prompt,out,batch) for _ in range(repeats)]
            times={key:round(statistics.median(d[key] for d in trials),3)
                   for key in ('prefill_ms','decode_ms','total_ms','decode_tokens_per_second')}
            cases.append(dict(architecture=name,parameters=parameters,prompt=prompt,
                              output=out,batch=batch,**times,
                              operations=trials[-1]['operations']))
    return cases


def bench_index():
    checks=[]
    for size in (2,16,256,4096,65536,262144):
        ix=PagedCausalIndex(2)
        for t in range(size):
            ix.append(t,0,0)
        pages=ix.page_count(0,0,size)
        for rank in (0,1,size//3,size//2,size-1):
            actual=ix.lookup_ordinal(0,0,size,rank)
            assert actual==size-1-rank
        iterations=2000
        started=time.perf_counter()
        for t in range(iterations):
            ix.lookup_page(0,0,size,0)
        recent_us=(time.perf_counter()-started)*1e6/iterations
        started=time.perf_counter()
        for t in range(iterations):
            ix.lookup_page(0,0,size, (t*17)%pages)
        historical_us=(time.perf_counter()-started)*1e6/iterations
        checks.append(dict(total_colliding_records=size,pages=pages,
                           deepest_page=pages-1,all_sampled_ranks_reachable=True,
                           recent_lookup_us=round(recent_us,3),
                           paged_lookup_us=round(historical_us,3)))
    return checks


def bench_programs():
    rng=np.random.default_rng(7)
    data=[]
    for entity_count in (256,4096):
        tables={r:rng.permutation(entity_count) for r in range(4)}
        for hops in (4,64,256):
            path=[i%4 for i in range(hops)]
            for count in (1,1024,65536):
                starts=rng.integers(0,entity_count,size=count)
                engine=VersionedProgramEngine(tables)
                started=time.perf_counter()
                serial, _ = engine.execute(starts,path,mode='serial')
                serial_ms=(time.perf_counter()-started)*1000
                started=time.perf_counter()
                compiled,info=engine.execute(starts,path,mode='compiled')
                compile_first_ms=(time.perf_counter()-started)*1000
                np.testing.assert_array_equal(serial,compiled)
                started=time.perf_counter()
                compiled_again,_=engine.execute(starts,path,mode='compiled')
                cached_ms=(time.perf_counter()-started)*1000
                np.testing.assert_array_equal(serial,compiled_again)
                data.append(dict(entities=entity_count,hops=hops,queries=count,
                                serial_vectorized_ms=round(serial_ms,4),
                                compile_and_apply_ms=round(compile_first_ms,4),
                                apply_cached_ms=round(cached_ms,4),
                                semantic_parity=True))
    return data


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=Path('v4_results.json'))
    p.add_argument('--repeats',type=int,default=3)
    args=p.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started=time.perf_counter()
    models=bench_models(args.repeats)
    indexed=bench_index()
    program=bench_programs()
    results={'environment':dict(python=platform.python_version(),
            torch=torch.__version__,numpy=np.__version__,threads=torch.get_num_threads(),
            repeats=args.repeats, model_weights='random/untrained',
            timing='median; fixed continuation; parity checks outside timed region'),
            'models':models,'indexed_reachability':indexed,
            'parallel_programs':program,'elapsed_seconds':round(time.perf_counter()-started,3)}
    args.output.write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps({'elapsed_seconds':results['elapsed_seconds'],
                      'model_cases':len(models),'index_cases':len(indexed),
                      'program_cases':len(program)}))

if __name__=='__main__':
    main()
