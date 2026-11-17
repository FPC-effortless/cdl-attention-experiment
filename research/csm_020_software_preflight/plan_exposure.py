"""Prospective exact sampler exposure accounting; no model fitting or DEV access."""
import argparse,json,random,hashlib
from pathlib import Path

def plan(prose,tasks,seed,updates=1024,batch=32,context=512):
 docs=[json.loads(x)['text'].encode() for x in Path(prose).read_text().splitlines()];docs=[x for x in docs if len(x)>context+1]
 rows=[json.loads(x) for x in Path(tasks).read_text().splitlines()];rng=random.Random(seed);events=[];counts=[];targets=positions=answers=0
 for step in range(updates):
  if step%2==0:
   samples=[]
   for _ in range(batch):
    di=rng.randrange(len(docs));offset=rng.randrange(len(docs[di])-context-1);samples.append({'article_index':di,'offset':offset})
   event={'update':step+1,'kind':'prose','samples':samples};t=p=batch*context
  else:
   chosen=[rows[rng.randrange(len(rows))] for _ in range(batch)]
   event={'update':step+1,'kind':'tasks','samples':[r['id'] for r in chosen]};t=sum(len(r['text'].encode('ascii'))-1 for r in chosen);p=batch*(max(len(r['text']) for r in chosen)-1);answers+=sum(sum(a['end']-a['start'] for a in r['answers']) for r in chosen)
  events.append(event);targets+=t;positions+=p;counts.append({'update':step+1,'targets':t,'padded_positions':p,'cumulative_targets':targets,'sequences':batch})
 return {'per_update':counts,'seed':seed,'updates':updates,'batch':batch,'targets':targets,'padded_positions':positions,'answer_targets':answers,'schedule_sha256':hashlib.sha256(json.dumps(events,sort_keys=True).encode()).hexdigest()}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--out',required=True);c=p.parse_args();root=Path(c.inputs)
 result=[plan(root/'prose_train_articles.jsonl',root/'train.jsonl',seed) for seed in [101,202,303]];Path(c.out).write_text(json.dumps(result,indent=2));print(json.dumps(result))
