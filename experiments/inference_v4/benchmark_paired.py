"""Interleaved paired V3/V4 cached inference benchmark, no correctness in timer.
Alternates trial order to reduce drift; untrained models on identical fixed
byte inputs, CPU one thread. Reports paired ratios AND spread, not a universal
performance claim.
"""
from pathlib import Path
import argparse, json, platform, statistics, time
import torch
from relational_composition.model import Config
from relational_composition.expressive_model import SerialMultiHopLM, StreamingMultiHop
from relational_composition.hierarchical_model import PagedMultiHopLM, StreamingPagedMultiHop


def once(model, engine, prompt, output, batch):
    torch.manual_seed(235)
    x=torch.randint(1,200,(batch,prompt))
    token=torch.full((batch,),33,dtype=torch.long)
    with torch.inference_mode():
        stream=engine(model)
        start=time.perf_counter()
        stream.prefill(x)
        prefill_ms=(time.perf_counter()-start)*1000
        start=time.perf_counter()
        for _ in range(output):
            stream.step(token)
        decode_ms=(time.perf_counter()-start)*1000
    return {'prefill_ms':prefill_ms,'decode_ms':decode_ms,'total_ms':prefill_ms+decode_ms}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repeats',type=int,default=8)
    p.add_argument('--output',default='paired_results.json')
    args=p.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    configs=[(128,256,1),(512,512,1),(256,256,4)]
    all_results=[]
    for prompt,output,batch in configs:
        torch.manual_seed(22)
        cfg=Config(width=24,layers=2,relations=4,addresses=8,candidates=2)
        models={'v3':(SerialMultiHopLM(cfg).eval(),StreamingMultiHop),
                'v4':(PagedMultiHopLM(cfg,page_bits=20).eval(),StreamingPagedMultiHop)}
        for name,(model,eng) in models.items():
            once(model,eng,16,8,batch)
        trials=[]
        for i in range(args.repeats):
            order=('v3','v4') if i%2==0 else ('v4','v3')
            result={}
            for name in order:
                result[name]=once(*models[name],prompt,output,batch)
            trials.append(result)
        ratios={field:[t['v4'][field]/t['v3'][field] for t in trials]
                for field in ('decode_ms','total_ms')}
        measurements={name:{field:statistics.median(t[name][field] for t in trials)
                   for field in ('prefill_ms','decode_ms','total_ms')}
                   for name in models}
        all_results.append(dict(prompt=prompt,output=output,batch=batch,
            measurements={name:{k:round(v,3) for k,v in metrics.items()}
                          for name,metrics in measurements.items()},
            paired_ratio={field:dict(median=round(statistics.median(values),4),
                      low=round(min(values),4),high=round(max(values),4))
                 for field,values in ratios.items()},
            decode_speed={name:round(batch*output*1000/measurements[name]['decode_ms'],2)
                          for name in models},
            complete_trial_timings=trials))
    record={'environment':dict(python=platform.python_version(),torch=torch.__version__,
             threads=1,weights='untrained',warmup=True,repeats=args.repeats,
             alternating_order=True,context='torch.inference_mode()'),
             'cases':all_results}
    Path(args.output).write_text(json.dumps(record,indent=2)+'\n')
    for row in all_results:
        print(row['prompt'],row['output'],row['batch'],row['decode_speed'],
              row['paired_ratio']['decode_ms'],flush=True)

if __name__=='__main__':main()
