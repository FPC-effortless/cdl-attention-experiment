"""Weight-free independent post-score recount. No imports from the study generator/model/scorer."""
from collections import Counter,defaultdict
from hashlib import sha256,file_digest
from io import BytesIO
from pathlib import Path
import argparse,ast,json,math,re

PANELS=('seen','novel','rewrite')
MODES=('bind','dated','latest')
ARMS=('answer','support'); SEEDS=(5371,5471,5571)
VALUES=set(); DATES=set()
HEADS=set(); NOUNS=set(); WRITERS=set()
PREFIX=('in the archive ','according to the ledger ')
SUFFIX=(' in the archive',' according to the ledger')
FACT=re.compile(r'(?:([a-z]+) filed|the record says ([a-z]+) listed) the ([a-z]+) ([a-z]+) as ([a-z]+)(?: on ([a-z]+))?')
ASK={
 'bind':re.compile(r'what color (?:is the|does the record list for the) ([a-z]+) ([a-z]+)'),
 'dated':re.compile(r'what color (?:was the|does the record list for the) ([a-z]+) ([a-z]+) on ([a-z]+)'),
 'latest':re.compile(r'what color (?:is the latest entry for the|does the latest record list for the) ([a-z]+) ([a-z]+)'),
}
def need(b,m):
 if not b:raise ValueError(m)
def raw(p,h=None):
 need(p.is_file() and not p.is_symlink(),'exact regular input')
 b=p.read_bytes();need(h is None or sha256(b).hexdigest()==h,'digest mismatch: '+p.name);return b
def js(p,h=None):
 def pairs(a):
  d=dict(a);need(len(d)==len(a),'duplicate keys');return d
 return json.loads(raw(p,h),object_pairs_hook=pairs)
def sha(p):
 need(p.is_file() and not p.is_symlink(),'digest regular file')
 with p.open('rb') as f:return file_digest(f,'sha256').hexdigest()
def frozen_lexicon(source):
 settings={}
 for n in ast.parse(source.decode()).body:
  if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('HEADS','NOUNS','WRITERS','VALUES','DATES'):
   k=n.targets[0].id;v=ast.literal_eval(n.value)
   need(isinstance(v,(tuple,list)) and all(isinstance(x,str) and re.fullmatch('[a-z]+',x) for x in v),'lexicon lists')
   settings[k]=set(v)
 need({k:len(v) for k,v in settings.items()}==dict(HEADS=8,NOUNS=8,WRITERS=3,VALUES=8,DATES=4),'exact new lexicon roster')
 for k,v in settings.items():globals()[k].update(v)
 return settings
def core(seg,isquery=False):
 if isquery:
  need(seg.endswith(' ?'),'question mark');seg=seg[:-2]
 left=next((r for r in PREFIX if seg.startswith(r)),'')
 if left:seg=seg[len(left):]
 right=next((r for r in SUFFIX if seg.endswith(r)),'')
 if right:seg=seg[:-len(right)]
 return seg,left,right
def fact(c):
 m=FACT.fullmatch(c);need(m is not None,'literal assertion grammar')
 a,b,k1,k2,v,d=m.groups();w=a or b
 need(w in WRITERS and k1 in HEADS and k2 in NOUNS and v in VALUES and (d is None or d in DATES),'literal assertion vocabulary')
 return (w,(k1,k2),v,d)
def query(c):
 hits=[(m,k.fullmatch(c)) for m,k in ASK.items() if k.fullmatch(c)]
 need(len(hits)==1,'literal query grammar')
 mode,r=hits[0];k1,k2,*tail=r.groups();date=tail[0] if tail else None
 need(k1 in HEADS and k2 in NOUNS and (date is None or date in DATES),'literal query vocabulary')
 return mode,(k1,k2),date
def parse(text,lex):
 parts=text.split(' . ')
 need(len(parts)==4 and all(0<len(s.split())<=24 and ' '.join(s.split())==s and set(s.split())<=lex for s in parts),'fixed visible boundaries/vocab')
 spans=tuple(core(z,i==3) for i,z in enumerate(parts))
 f=tuple(fact(z[0]) for z in spans[:3]);q=query(spans[-1][0]);mode,key,date=q
 need(len({x[0] for x in f})==3 and all((x[-1] is not None)==(mode=='dated') for x in f),'exact uniquely attributed date convention')
 candidates=[i for i,x in enumerate(f) if x[1]==key and (mode!='dated' or x[3]==date)]
 need(len(candidates)==(2 if mode=='latest' else 1),'answer identifiable in constant scope')
 i=candidates[-1];return dict(f=f,q=q,target=i,gold=f[i][2],spans=spans,tokens=len(text.split()))
def validate_family(rows,t,lex,panel):
 need(isinstance(rows,list) and len(rows)==len(set(rows))==4 and set(t)=={'mode','gold','target'},'family wire')
 p=[parse(x,lex) for x in rows];m=p[0]['q'][0]
 need(t['mode']==m and all(z['q'][0]==m for z in p) and
      t['gold']==[z['gold'] for z in p] and t['target']==[z['target'] for z in p],'independently derived mode/label/target index')
 f=[z['f'] for z in p];q=[z['q'] for z in p]
 n=next((j for j in range(3) if f[0][j]==f[1][j]),None)
 need(n is not None and all(f[0][n]==v[n] for v in f),'fixed identical distractor and slot')
 rel=[tuple(x for i,x in enumerate(z) if i!=n) for z in f]
 need(rel[0]==rel[1][::-1] and rel[2]==rel[3][::-1] and q[0]==q[1] and q[2]==q[3],'two presentation mirrors, same query')
 need(f[0][n][1] not in {x[1] for x in rel[0]+rel[2]},'wrong-key distractor')
 gu=[z['gold'] for z in p];need(len(set(gu))==2,'must require two distinct golds')
 if m in ('bind','dated'):
  need(f[0]==f[2] and f[1]==f[3] and q[0]!=q[2] and gu[0]==gu[1] and gu[2]==gu[3] and gu[0]!=gu[2],'invariant order/directional opposite probes')
  if m=='bind':
   need({q[0][1],q[2][1]}=={r[1] for r in rel[0]} and q[0][1]!=q[2][1],'both keys queried')
  else:
   need(q[0][1]==q[2][1] and q[0][2]!=q[2][2] and all(x[1]==q[0][1] for x in rel[0]),'two exact dates queried for one key')
 else:
  need(len(set(q))==1 and all(x[1]==q[0][1] for x in rel[0]+rel[2]) and
       [x[0:2]+x[3:] for x in rel[0]]==[x[0:2]+x[3:] for x in rel[2]] and
       [x[2] for x in rel[0]]==[x[2] for x in rel[2]][::-1] and
       gu[0]==gu[3] and gu[1]==gu[2] and gu[0]!=gu[1],'latest gold flips on order and payload independently')
 for i,z in enumerate(p):
  for c,l,r in z['spans']:
   need((bool(l) and bool(r)) if panel in ('novel','rewrite') else (not l or not r),'exact wording regime')
 return p

def panel(rows,labels,name,lex,cross):
 exp={'train':512,'cal':128,'seen':256,'novel':256,'rewrite':256}[name]
 need(len(rows)==len(labels)==3*exp,'exact cohort bounds')
 modes=Counter();keys=set();out=[];slots=defaultdict(Counter);colors=defaultdict(Counter);rawseen=set()
 for family,t in zip(rows,labels,strict=True):
  need(not rawseen.intersection(family) and not cross.intersection(family),'exact prompt duplication across cohort')
  rawseen.update(family);p=validate_family(family,t,lex,name);m=t['mode'];modes[m]+=1
  out.append(p);keys.update(x[1] for z in p for x in z['f'])
  slots[m].update(str(z['target']) for z in p);colors[m].update(z['gold'] for z in p)
 need(modes=={m:exp for m in MODES} and len(keys)==16 and {k[0] for k in keys}==HEADS and {k[1] for k in keys}==NOUNS,'mode and lexical support')
 need(all(set(v)==set('12' if m=='latest' else '012') for m,v in slots.items()) and all(set(v)==VALUES and len(set(v.values()))==1 for v in colors.values()),'target slot support and exact color marginal balance')
 cross.update(rawseen)
 return out,keys,dict(families=3*exp,rows=12*exp,keys=len(keys),mode_families=modes,target_slots=slots,gold_colors=colors,
                       tokens=[min(z['tokens'] for f in out for z in f),max(z['tokens'] for f in out for z in f)])
def check_rewrite(a,b):
 ch=Counter()
 for fam,new in zip(a,b,strict=True):
  for x,y in zip(fam,new,strict=True):
   need(all(x[k]==y[k] for k in ('q','f','gold','target')),'rewrite is exact same visible semantics')
   for (c,l,r),(cc,ll,rr) in zip(x['spans'],y['spans'],strict=True):
    need(c==cc and ll==(l or PREFIX[0]) and rr==(r or SUFFIX[1]),'exact known-component rewrite')
    ch['added_prefix']+=not l;ch['added_suffix']+=not r
 return dict(ch)

def decode(torch,p,h,n,width,kind):
 b=raw(p,h);need(0<len(b)<=8_000_000,'bounded prediction wire')
 need(not torch.serialization.get_unsafe_globals_in_checkpoint(BytesIO(b)),'unsupported predicted-wire global')
 x=torch.load(BytesIO(b),weights_only=True,map_location='cpu')
 shapes=(dict(logits=(n,width),weights=(n,3),hard=(n,width),q=(n,48),v=(n,3,48))
          if kind=='primary' else dict(gold=(n,width)))
 need(type(x)is dict and set(x)==set(shapes),'exact prediction output key set')
 for nm,sh in shapes.items():
  t=x[nm];need(type(t)is torch.Tensor and t.dtype==torch.float32 and t.device.type=='cpu' and t.is_contiguous() and tuple(t.shape)==sh
               and not t.requires_grad and bool(torch.isfinite(t).all()),'finite exact wire geometry')
 return {k:x[k].tolist() for k in ('logits','weights','hard') if k in x}|({'gold':x['gold'].tolist()} if kind=='gold' else {})

def compare_matrix(families,native,attention,hard,gold,ids):
 types=('soft','own_hard') if gold is None else ('soft','own_hard','gold_hard')
 by={m:dict(n=0,top1_ties=0,actor_correct_rows=0,strict_top1_families=0,unique_correct_rows=0,strict_unique_families=0,
            gold_parity_checks=0,methods={k:dict(row_correct=0,strict=0,answer_ties=0,
                table_unique_top1_x_answer={i:0 for i in ('0:0','0:1','1:0','1:1')},
                table_operational_top1_x_answer={i:0 for i in ('0:0','0:1','1:0','1:1')},
                table_all4_unique_support_x_all4_answer={i:0 for i in ('0:0','0:1','1:0','1:1')}) for k in types}) for m in MODES}
 opts={}
 outs={k:[] for k in types}
 for i,fam in enumerate(families):
  mode=fam[0]['q'][0]; c=by[mode];c['n']+=1; actor=[];unique=[]; got={k:[] for k in types}; best={k:[] for k in types}
  for j,row in enumerate(fam):
   r=4*i+j;w=attention[r];mx=max(w);ix=w.index(mx); actual=ix==row['target'];uq=actual and w.count(mx)==1
   actor.append(actual);unique.append(uq)
   c['top1_ties']+=w.count(mx)>1;c['unique_correct_rows']+=uq;c['actor_correct_rows']+=actual
   seq=[native[r],hard[r]] if gold is None else [native[r],hard[r],gold[r]]
   if gold is not None and actual:
    need(all(abs(x-y)<=2e-5+1e-6*abs(y) for x,y in zip(hard[r],gold[r],strict=True)),
         'gold-hard and own-hard must agree for an identical selected clause')
    c['gold_parity_checks']+=1
   for k,z in zip(types,seq,strict=True):
    top=max(z);a=z.index(top) if z.count(top)==1 else None;ok=a==ids[row['gold']]
    v=c['methods'][k];v['row_correct']+=ok;v['answer_ties']+=a is None
    v['table_unique_top1_x_answer'][f'{int(uq)}:{int(ok)}']+=1
    v['table_operational_top1_x_answer'][f'{int(actual)}:{int(ok)}']+=1
    got[k].append(ok);best[k].append(a)
  c['strict_top1_families']+=all(actor);c['strict_unique_families']+=all(unique)
  for k in types:
   c['methods'][k]['strict']+=all(got[k])
   c['methods'][k]['table_all4_unique_support_x_all4_answer'][f'{int(all(unique))}:{int(all(got[k]))}']+=1
   outs[k].append((mode,tuple(got[k]),tuple(best[k])))
 for m,c in by.items():
  n=c['n']
  need(n==256 and all(sum(v['table_unique_top1_x_answer'].values())==4*n
      and sum(v['table_operational_top1_x_answer'].values())==4*n
      and sum(v['table_all4_unique_support_x_all4_answer'].values())==n for v in c['methods'].values()),'exact 2x2 table denominators')
  c['unique_support_gate']=100*c['unique_correct_rows']>=95*4*n and 100*c['strict_unique_families']>=90*n
 cmp={f'{m}_vs_soft':change(outs['soft'],outs[m]) for m in types if m!='soft'}
 if 'gold_hard' in types:cmp['gold_hard_vs_own_hard']=change(outs['own_hard'],outs['gold_hard'])
 return by,cmp,outs
def judge(families,L,W,vocab):
 out={m:Counter() for m in MODES};ds={m:Counter() for m in MODES};flags=[]
 for i,rows in enumerate(families):
  m=rows[0]['q'][0];b=out[m];x=ds[m];b['n']+=1; ok=[];sel=[]; tops=[]; uniq=[]
  for j,row in enumerate(rows):
   scores=L[4*i+j];att=W[4*i+j]; k=max(scores);s=max(att)
   need(all(0<=v<=1 for v in att) and abs(math.fsum(att)-1)<=1.2e-6,'nonnegative normalized read weights')
   top=scores.index(k) if scores.count(k)==1 else None;tops.append(top)
   ok.append(top==vocab[row['gold']]); sel.append(att.index(s)==row['target'])
   uniq.append(att.count(s)==1 and sel[-1])
   x['answer_ties']+=top is None;x['selection_ties']+=att.count(s)!=1
   x['correct_without_unique_target_selection']+=ok[-1] and not uniq[-1]
   x['incorrect_with_unique_target_selection']+=not ok[-1] and uniq[-1]
  b['correct']+=sum(ok);b['strict']+=all(ok);b['selected']+=sum(sel);b['selection_strict']+=all(sel)
  x['unique_target_selected']+=sum(uniq);x['unique_target_selection_strict']+=all(uniq)
  for a,c in [(0,1),(2,3)]:
   x['order_pairs_strict']+=ok[a] and ok[c]
   x['order_pairs_one_correct']+=ok[a]!=ok[c]
   both=tops[a] is not None and tops[c] is not None
   aligned=both and ((tops[a]!=tops[c]) if m=='latest' else (tops[a]==tops[c]))
   x['order_pairs_behavior_aligned']+=aligned
   x['order_pairs_aligned_but_not_both_correct']+=aligned and not(ok[a] and ok[c])
  x['opposite_probe_same_order_strict']+=(ok[0] and ok[2])+(ok[1] and ok[3])
  flags.append((m,tuple(ok),tuple(tops)))
 return {m:dict(v) for m,v in out.items()},{m:dict(v) for m,v in ds.items()},flags
def passes(d,n):
 need(set(d)==set(MODES) and all(v['n']==n for v in d.values()),'complete gate denominators')
 return all(v['correct']*100>=95*4*n and v['strict']*100>=90*n for v in d.values())
def change(a,b):
 by={m:Counter() for m in MODES}
 for (m,x,tx),(n,y,ty) in zip(a,b,strict=True):
  need(m==n,'paired mode order')
  c=by[m];c['n']+=1;c['family_rescues']+=not all(x) and all(y);c['family_damage']+=all(x) and not all(y)
  c['row_rescues']+=sum(not o and r for o,r in zip(x,y,strict=True))
  c['row_damage']+=sum(o and not r for o,r in zip(x,y,strict=True));c['top_changes']+=sum(o!=r for o,r in zip(tx,ty,strict=True))
 return {m:dict(v) for m,v in by.items()}
def main():
 ap=argparse.ArgumentParser()
 for x in ('data','results','out'):ap.add_argument('--'+x,type=Path,required=True)
 ap.add_argument('--with-gold',action='store_true')
 for x in ('custody-sha','run-sha','generator-sha','result-sha'):ap.add_argument('--'+x,required=True)
 a=ap.parse_args(); root=Path(__file__).resolve().parent
 need(sha(root/'run.py')==a.run_sha and sha(root/'data_gen.py')==a.generator_sha,'exact reviewed study sources')
 lex_source=frozen_lexicon(raw(root/'data_gen.py',a.generator_sha))
 p=js(a.data/'custody.json',a.custody_sha)
 names={'train.json','cal.json','lexicon.json'}|{f'{n}-{x}.json' for n in PANELS for x in ('prompts','truth')}
 need(set(p)==names and {f.name for f in a.data.iterdir()}==names|{'custody.json'},'exact data closure')
 lex=js(a.data/'lexicon.json',p['lexicon.json'])
 need(lex==sorted(set(lex)) and set(lex).issuperset(HEADS|NOUNS|VALUES|DATES|WRITERS|{'.','?'}),'fixed sorted lexicon')
 ids={x:y for y,x in enumerate(lex)}
 ds={};ks={};sc={};seen=set()
 for n in ('train','cal')+PANELS:
  if n in ('train','cal'):
   t=js(a.data/(n+'.json'),p[n+'.json']);need(set(t)=={'text','truth'},'train/cal wire'); rows,lab=t['text'],t['truth']
  else:
   rows,lab=(js(a.data/f'{n}-{kind}.json',p[f'{n}-{kind}.json']) for kind in ('prompts','truth'))
  ds[n],ks[n],sc[n]=panel(rows,lab,n,set(lex),seen)
 need(all(not ks[n]&ks[m] for i,n in enumerate(('train','cal','seen','novel')) for m in ('train','cal','seen','novel')[i+1:])
      and ks['seen']==ks['rewrite'],'exact lexical separation and rewrite match')
 rew=check_rewrite(ds['seen'],ds['rewrite'])
 gold=js(a.results/'RESULTS.json',a.result_sha)
 expected={f'{arm}-{seed}' for arm in ARMS for seed in SEEDS}
 by={f"{r['arm']}-{r['seed']}":r for r in gold}
 need(set(by)==expected and len(gold)==6,'six fixed arm-seed outputs')
 import torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 reports={};top={};method_flags={}
 for arm in ARMS:
  for seed in SEEDS:
   n=f'{arm}-{seed}';dp=a.results/n; orig=by[n]
   f=js(dp/'fit.json'); prem=js(dp/'predict.json')
   need((f['arm'],f['seed'],f['steps'])==(arm,seed,1024) and f['source_sha']==a.run_sha and f['generator_sha']==a.generator_sha
     and f['train_sha']==p['train.json'] and f['cal_sha']==p['cal.json'] and sha(dp/'weights.pt')==f['weight_sha']
     and len(f['losses'])==1024 and all(math.isfinite(x) and x>=0 for x in f['losses'])
     and math.isfinite(f['fit_seconds']) and f['fit_seconds']>0,'bound realized fit evidence, no weight decoding')
   need(set(prem)==set(PANELS) and all(orig[k]==v for k,v in f.items() if k!='losses'),'fixed primary roster and source')
   need(all(set(x)=={'n','correct','strict','selected','selection_strict'} and 0<=x['strict']<=x['n'] and 0<=x['correct']<=4*x['n']
     for x in f['cal'].values()),'retained TRAIN/Cal counts')
   cal=passes(f['cal'],128);need(cal==f['adequacy']==orig['adequacy'],'calibration adequacy flags')
   rr={};top[n]={};method_flags[n]={}
   gm=js(dp/'gold.json') if a.with_gold else None
   need(gm is None or set(gm)==set(PANELS),'exact gold manifest panels')
   for m in PANELS:
    h=prem[m];need(set(h)=={'wire_sha','prompts_sha'} and h['prompts_sha']==p[m+'-prompts.json'],'pinned predictions/order')
    wire=decode(torch,dp/(m+'.pt'),h['wire_sha'],3072,len(lex),'primary'); z,w=wire['logits'],wire['weights']
    gx=None
    if gm is not None:
     v=gm[m];need(set(v)=={'primary_sha','gold_sha','truth_sha'} and v['primary_sha']==h['wire_sha'] and v['truth_sha']==p[m+'-truth.json'],
       'gold diagnostic bound to exact primary and same derivable truth')
     gx=decode(torch,dp/(m+'-gold.pt'),v['gold_sha'],3072,len(lex),'gold')['gold']
    primary,diag,top[n][m]=judge(ds[m],z,w,ids)
    matrices,moves,method_flags[n][m]=compare_matrix(ds[m],z,w,wire['hard'],gx,ids)
    need(all(matrices[k]['methods']['soft']['row_correct']==primary[k]['correct'] and
             matrices[k]['methods']['soft']['strict']==primary[k]['strict'] and
             matrices[k]['actor_correct_rows']==primary[k]['selected'] and
             matrices[k]['strict_top1_families']==primary[k]['selection_strict'] for k in MODES),'primary vs mechanism join')
    need(primary==orig[m],'independent logits and target-index score mismatch')
    rr[m]=dict(primary=primary,diagnostics=diag,mechanism_2x2=matrices,paired_read_changes=moves,
          panel_pass=passes(primary,256),wire_sha256=h['wire_sha'],
          gold_sha256=(gm[m]['gold_sha'] if gm is not None else None))
   held=all(rr[x]['panel_pass'] for x in ('seen','novel'))
   need(held==orig['held_pass'] and (held and cal)==orig['qualified'],'exact frozen final qualification')
   support=all(rr[nn]['mechanism_2x2'][k]['unique_support_gate'] for nn in ('seen','novel') for k in MODES)
   reports[n]=dict(panels=rr,calibration=f['cal'],adequacy=cal,held_pass=held,qualified=cal and held,support_gate=support,
      gold_manifest_sha256=sha(dp/'gold.json') if gm is not None else None,
      init=f['common_init'],params=f['params'],fit_process_wall_seconds=f['fit_seconds'],fit_sha256=sha(dp/'fit.json'),
      predict_sha256=sha(dp/'predict.json'),rewrite_vs_seen=change(top[n]['seen'],top[n]['rewrite']))
 matched={}
 for seed in SEEDS:
  x,y=f'answer-{seed}',f'support-{seed}'
  need(all(reports[x][k]==reports[y][k] for k in ('init','params')),'same nominal parameters and paired initialization')
  matched[str(seed)]={p:{m:change(method_flags[x][p][m],method_flags[y][p][m])
                              for m in method_flags[x][p]} for p in PANELS}
 res=dict(status='INDEPENDENT_BIND_001A_RECOUNT',source_sha256=sha(Path(__file__)),run_sha256=a.run_sha,generator_sha256=a.generator_sha,
    result_sha256=a.result_sha,custody_sha256=a.custody_sha,vocabulary=len(lex),data=sc,rewrite_intervention=rew,matched_support_vs_answer=matched,
    fits=reports,gold_diagnostic_included=a.with_gold,model_calls=0,fit_weights_deserialized=False,qualification_recalculated=True,
    limitations='Only retained calibration count gates; no calibration logits or optimizer trace. Target attention ranking is diagnostic, not causal attribution. Rewrite not gating.')
 with a.out.open('x') as f:json.dump(res,f,allow_nan=False,sort_keys=True,separators=(',',':'));f.write('\n')
 a.out.chmod(0o400);print(sha(a.out))
if __name__=='__main__':main()
