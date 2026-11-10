"""Post-fit descriptive local-context lookup; not a matched learned baseline."""
from collections import Counter,defaultdict
import json,math
from run import HERE,corpus

def main():
 tr,_=corpus();raw=tr[:129].tolist();rows=[]
 for width in (1,2,3):
  counts=defaultdict(Counter)
  for t in range(128):
   context=tuple(([-1]*width+raw[:t+1])[-width:]);counts[context][raw[t+1]]+=1
  correct=sum(max(c.values()) for c in counts.values());nll=0
  for c in counts.values():
   n=sum(c.values());nll-=sum(v*math.log(v/n) for v in c.values())
  rows.append(dict(context_bytes=width,unique_contexts=len(counts),training_lookup_accuracy=correct/128,empirical_conditional_nll=nll/128))
 result=dict(scope='Post-fit descriptive lookup on the same training excerpt; no generalization or matched-budget claim',rows=rows)
 (HERE/'local_context_control.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
