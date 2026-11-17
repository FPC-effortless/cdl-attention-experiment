import random,json
from pathlib import Path
POS=[('is','code'),('is','secret'),('set','code'),('set','secret'),('update','secret')]
NEG=[('is','marker'),('is','tag'),('set','marker'),('update','marker'),('update','tag')]
HOLD=[('update','code'),('set','tag')]
def clause(verb,role,value):return f'The {role} is {value}.' if verb=='is' else f'{verb.capitalize()} {role} to {value}.'
def make(seed,count,pos,neg,order,gap):
 rng=random.Random(seed);pairs=[]
 for i in range(count):
  a=i%4;b=(a+1+(i//16)%3)%4;other=[v for v in range(4) if v not in[a,b]];old=rng.choice(other);noise=(i//4)%4;nonce=''.join(rng.choice('jklmnopq') for _ in range(10));prefix=[];labels=[]
  for target in [a,b]:
   text='Note '+nonce+'. ';ev=[0]*len(text)
   ops=[(('is','code'),old,True),(neg,noise,False),(pos,target,True)] if order=='pre' else [(('is','code'),old,True),(pos,target,True),(neg,noise,False)]
   for (verb,role),v,positive in ops:
    c=clause(verb,role,chr(97+v));e=[0]*len(c)
    if positive:e[-2]=v+1
    filler=(' rain sun wind. '*3)[:gap//2];text+=c+' '+filler;ev+=e+[0]*(1+len(filler))
   prefix.append(text);labels.append(ev)
  suffix=(' rain sun wind. '*12)[:gap]+' The current code is '
  pairs.append({'group':i,'prefix':prefix,'suffix':suffix,'targets':[97+a,97+b],'events':labels,'positive_form':list(pos),'negative_form':list(neg),'order':order})
 return pairs
def build():
 train={};i=0
 for p in POS:
  for n in NEG:
   for order in ['pre','post']:
    for gap in [0,16,32]:
     train[f'{p[0]}_{p[1]}-{n[0]}_{n[1]}-{order}-{gap}']=make(41000+i*31,32,p,n,order,gap);i+=1
 specs=[('familiar',('set','code'),('update','marker'),'post',32),('held_positive',('update','code'),('is','marker'),'post',32),('held_negative',('set','secret'),('set','tag'),'post',32),('held_both',('update','code'),('set','tag'),'post',32),('held_both_pre',('update','code'),('set','tag'),'pre',32),('long',('update','code'),('set','tag'),'post',128)]
 dev={name:make(61000+i*101,64,p,n,order,gap) for i,(name,p,n,order,gap) in enumerate(specs)}
 return {'train':train,'dev':dev}
if __name__=='__main__':(Path(__file__).parent/'data.json').write_text(json.dumps(build()))
