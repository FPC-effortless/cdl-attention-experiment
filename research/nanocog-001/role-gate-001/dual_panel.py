"""Audit exact sealed ROLE-002/003 predictions without executing a model."""
from collections import Counter
from hashlib import sha256
import itertools,json,math
from pathlib import Path
from statistics import mean
from trainlab.runner import verify_run
ROOT=Path(__file__).resolve().parent.parent
DATA={
 '002':dict(ledger='2f28213c2eedcbdfce2151a42250ba57b0ca48bd92a2119726dc10818679ad25',seal='e6a31a5ceaecb1ee296289f4f38a63d541f44de6e0508c02ad55ac9542e056f8',actor='201a63921dfaaf601b03e03e7dc80858e9958b71fe0eef8e6b8572d8d459aac1',judge='ce80c30bfbb9c38ae8b078d16c0fd8187bc8a641bba4315bd05dad5253927118',result='a6c354d69ee3ac3583cab31125404987fe510c22e3e282d93325ab0a04b390cb',arm='w025'),
 '003':dict(ledger='d9b6bd3cd2a9a089933c51a14383bc3378d88e837d9d5427e81175d22a826d48',seal='a03f10d2ec6b2ef1fee936c031dad2f17ab6d7499f61cc6fb761bb1c72321d12',actor='536ba918ab5122549cd1142df0c02789fb406731a2fdf4928db75ac04a1c1555',judge='edade4df767fff4ce0321d09c4e5511514f3944e8ea5f52077539402b7f4de84',result='8f80309fb58d1f39a7533e8aeb3b28305957c4894e0ccee154470f25fc8b43d2',arm='clip1'),
}
def req(x,s):
 if not x:raise ValueError(s)
def read(p,d=None):
 b=p.read_bytes();req(d is None or sha256(b).hexdigest()==d,'digest: '+str(p))
 return json.loads(b)
def winner(x,y):
 return 'first' if x>y else 'second' if x<y else 'tie'
def main():
 out={}; cx={}
 for stag,reg in DATA.items():
  p=ROOT/f'cog-role-{stag}'
  led=read(p/'actor-admissions/ledger.json',reg['ledger']); seal=read(p/'PREDICTION_SEAL.json',reg['seal'])
  off=read(p/'RESULTS.json',reg['result'])
  prompts=[p.split() for p in read(p/'panel/confirm_actor.json',reg['actor'])['prompts']]
  fs=read(p/'panel/confirm_judge.json',reg['judge'])['families']
  req(len(prompts)==960 and len({tuple(z) for z in prompts})==960 and len(fs)==120,'panel size')
  cx[stag]=set(map(tuple,prompts))
  for seed in (1171,1271,1371):
   cid=f"{reg['arm']}-{seed}-1280";c=led[cid];s=seal[cid];run=p/'actors'/s['id']
   rr=read(run/'run.json',s['receipt_sha256'])
   req(verify_run(run)['valid'] and rr['status']=='succeeded' and rr['name']==cid and rr['inputs']==c['inputs'] and rr['artifacts']['predictions.json']==s['prediction_sha256'],'native')
   raw=read(run/'output/predictions.json',s['prediction_sha256'])
   req(raw['checkpoint_sha256']==c['checkpoint_sha256'] and raw['actor_sha256']==reg['actor'] and raw['seed']==seed and raw['updates']==1280 and raw['arm']=='gated32','checkpoint')
   vocab=read(p/'actor-admissions'/cid/'vocabulary.json',c['inputs']['vocabulary.json'])
   req(vocab['size']==raw['vocab_size']==287 and vocab['words']==raw['tokenizer_words'],'vocab')
   ix={w:i+262 for i,w in enumerate(vocab['words'])}
   req(len(raw['logits'])==960,'logit coverage')
   fam_complete=exact=0; slot=Counter(); qpair=Counter(); opair=Counter(); cell_quality=Counter()
   nll=[]; duplicate_pairs=0
   for i,f in enumerate(fs):
    data=prompts[i*8:(i+1)*8];outcomes=[]
    req(f['rows']==list(range(i*8,i*8+8)),'row map')
    for j,z in enumerate(data):
     a,is1,x,b,is2,y,q=z; req(is1==is2=='is' and a!=b and x!=y and q in (a,b),'prompt')
     vals=raw['logits'][8*i+j];req(len(vals)==287 and all(math.isfinite(z) for z in vals),'logits')
     truth={a:x,b:y}[q];req(truth==f['answers'][j],'annotation')
     t=ix[truth];u=ix[x];v=ix[y]
     selected=winner(vals[u],vals[v]); code='first' if q==a else 'second'
     strict=all(z<vals[t] for n,z in enumerate(vals) if n!=t)
     color=selected==code
     slot[(code,'den')]+=1;slot[(code,'strict')]+=strict;slot[(code,'color')]+=color
     zmax=max(vals)
     masses=[math.exp(w-zmax) for w in vals]
     pairmass=(masses[u]+masses[v])/sum(masses)
     nll.append((-math.log(masses[t]/(masses[u]+masses[v])),-math.log(pairmass)))
     outcomes.append((selected,code,truth,strict,color,x if selected=='first' else y if selected=='second' else 'tie'))
    for j in (0,2,4,6):
     a,b=outcomes[j],outcomes[j+1]
     req(data[j][:-1]==data[j+1][:-1] and a[2]!=b[2],'query edge')
     k=('tie' if 'tie' in (a[0],b[0]) else
        'always_'+a[0] if a[0]==b[0] else
        'follows_query' if a[4] and b[4] else 'inverted_query')
     qpair[k]+=1
    for j in (0,1,4,5):
     a,b=outcomes[j],outcomes[j+2]
     req(data[j][:3]==data[j+2][3:6] and data[j][3:6]==data[j+2][:3] and a[2]==b[2],'order edge')
     k=('tie' if 'tie' in (a[0],b[0]) else
        'always_'+a[0] if a[0]==b[0] else
        'follows_role' if a[4] and b[4] else 'stable_wrong_role')
     opair[k]+=1
    exact+=sum(r[3] for r in outcomes);fam_complete+=all(r[3] for r in outcomes)
   req(exact==off[cid]['all']['exact_worlds'] and fam_complete==off[cid]['all']['exact_families'],'official totals')
   req(sum(qpair.values())==sum(opair.values())==480,'edge accounting')
   out[f"{stag}-{seed}"]=dict(checkpoint_sha256=c['checkpoint_sha256'],strict_worlds=exact,strict_families=fam_complete,
    queried_slot={k:{z:slot[k,z] for z in ('den','strict','color')} for k in ('first','second')},query_pairs=dict(qpair),
    order_pairs=dict(opair),mean_choice_nats=mean(a for a,b in nll),mean_pair_mass_nats=mean(b for a,b in nll))
 req(not(cx['002']&cx['003']),'panel prompts overlap')
 req(all(out[f'002-{s}']['checkpoint_sha256']==out[f'003-{s}']['checkpoint_sha256'] for s in (1171,1271,1371)),'frozen checkpoints differ')
 print(json.dumps(out,indent=2,allow_nan=False,sort_keys=True))
if __name__=='__main__':main()
