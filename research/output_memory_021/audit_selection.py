"""Frozen prospective TRAIN subset identity/coverage audit; no model outputs."""
import argparse,hashlib,json,re
from collections import Counter
from pathlib import Path
from surface_patterns import PATTERNS,audit_surface

def audit(path):
 raw=Path(path).read_bytes();rows=[json.loads(x) for x in raw.splitlines()]
 if len(rows)!=8192:raise ValueError('original8192TRAINsessions required')
 selected=rows[:4096];pairs={}
 for r in selected:
  if not r['id'].startswith('train-'):raise ValueError('TRAIN only')
  audit_surface(r);pairs.setdefault(r['pair_id'],[]).append(r)
 if len(pairs)!=2048 or any(len(v)!=2 or {r['variant'] for r in v}!={0,1} for v in pairs.values()):raise ValueError('complete paired roster')
 serialized=''.join(json.dumps(r,sort_keys=True)+'\n' for r in selected).encode()
 counts=lambda x:dict(sorted(Counter(x).items()))
 templates=Counter();payload=[Counter() for _ in range(3)];role={}
 for r in selected:
  for o in r['operations']:
   text=r['text'][o['start']:o['end']].strip()
   matches=[i for i,pattern in enumerate(PATTERNS[o['op']]) if re.fullmatch(pattern,text)]
   if len(matches)!=1:raise ValueError('surface template')
   templates[f"{o['op']}:{matches[0]}"]+=1
   for k in ('dst','src'):
    if o.get(k) is not None:role.setdefault(o['op']+':'+k,Counter())[o[k]]+=1
   if o['op']=='set':
    for i,b in enumerate(o['value']):payload[i][b]+=1
  for a in r['answers']:
   text=r['text'][a['question_start']:a['start']].removesuffix('\nAnswer: ').strip()
   matches=[i for i,p in enumerate(PATTERNS['query']) if re.fullmatch(p,text)]
   if len(matches)!=1:raise ValueError('query template')
   templates[f'query:{matches[0]}']+=1
 ids=[r['id'] for r in selected]
 return {'selection_version':'first2048-complete-train-pairs-v1','selection_rule':'first4096sessions in frozen8192file, retainingbothvariants; no outcome selection','original_sha256':hashlib.sha256(raw).hexdigest(),'selected_sha256':hashlib.sha256(serialized).hexdigest(),'selected_serialization':'json.dumps(record,sort_keys=True) + newline, UTF8','selected_id_roster_sha256':hashlib.sha256('\n'.join(ids).encode()).hexdigest(),'sessions':len(selected),'pairs':len(pairs),'queries':sum(len(r['answers']) for r in selected),'selected_pair_ids':list(pairs),'selected_session_ids':ids,'coverage':{'requested_delay_sessions':counts(r['requested_delay_bytes'] for r in selected),'actual_delay_sessions':counts(r['delay_bytes'] for r in selected),'wording_metadata':counts(r['wording'] for r in selected),'surface_template_counts':dict(sorted(templates.items())),'name_bank':counts(r['names'] for r in selected),'identity_roles':{k:dict(sorted(v.items())) for k,v in role.items()},'SET_payload_position_char_counts':[dict(sorted(v.items())) for v in payload],'query_kinds':counts(a['kind'] for r in selected for a in r['answers'])},'scope':'known finiteTRAIN recipe, not comprehensive linguistic coverage; repetition increases under fixed optimizer exposure; remaining4096TRAINsessions unused, not reclassified held-outTEST'}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True);a=p.parse_args();result=audit(a.data);Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ('selected_pair_ids','selected_session_ids','coverage')},indent=2))
