"""Read-only audit of archived calibration sufficient statistics, not retraining."""
import argparse, collections, hashlib, json, math, pathlib, statistics
NAMES=('cog','dense','gated','gru','static')
SEEDS=(171,311,591)
METHODS=('pooled','age-stratified','conservative-age','pack-max')
REGIMES=('0.15-0','0.15-1','0.5-0','0.5-1')
H=(0,5,15)
FIELDS=('rmse_v','mae_v','coverage90','width_v','bias_v')
STRATA={'age':('0-3','4-15','16+'),'last-observed-voltage':('low','mid','high'),'last-observed-current':('low','mid','high')}
def sha(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def verify_manifest(root):
 lines=(root/'SHA256SUMS.txt').read_text().splitlines()
 paths=set()
 for line in lines:
  digest,path=line.split('  ',1);p=root/path
  assert p.is_file() and p.resolve().is_relative_to(root.resolve()) and path not in paths
  assert sha(p)==digest,path;paths.add(path)
 assert {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name!='SHA256SUMS.txt'}=={str(pathlib.Path(p)) for p in paths}
 return len(paths)
def aggregate(rows):
 n=sum(r['n'] for r in rows)
 if not n:return {'n':0}
 return {'n':n,'rmse_v':math.sqrt(sum(r['sse'] for r in rows)/n),'mae_v':sum(r['sae'] for r in rows)/n,'coverage90':sum(r['covered'] for r in rows)/n,'width_v':sum(r['width'] for r in rows)/n,'bias_v':sum(r['signed_error'] for r in rows)/n}
def check(a,b):
 assert a['n']==b['n']
 if not a['n']:return
 for f in FIELDS:
  assert math.isfinite(a[f]) and abs(a[f]-b[f])<2e-9,(f,a,b)
def ev(e,methods,test,expected_counts):
 rows=collections.defaultdict(list);strata=collections.defaultdict(list)
 for r in e['flight_metrics']:
  assert r['flight'] in test
  rows[(r['method'],r['regime'],r['horizon'])].append(r)
 for r in e['flight_strata']:
  assert r['flight'] in test
  strata[(r['method'],r['regime'],r['horizon'],r['group'],r['bin'])].append(r)
 for m in methods:
  for r in REGIMES:
   for h in H:
    key=(m,r,h);rr=rows[key];assert len(rr)==len(test) and {x['flight'] for x in rr}==set(test)
    for x in rr:
     k=(x['flight'],r,h)
     if k in expected_counts:assert x['n']==expected_counts[k]
     else:expected_counts[k]=x['n']
    check(aggregate(rr),e['test']['/'.join(map(str,key))])
    counts={x['flight']:x['n'] for x in rr}
    for g,bins in STRATA.items():
     totals=collections.Counter()
     for b in bins:
      kk=(*key,g,b);ss=strata[kk]
      assert len(ss)==len(test) and {x['flight'] for x in ss}==set(test)
      check(aggregate(ss),e['strata']['/'.join(map(str,kk))])
      for x in ss:totals[x['flight']]+=x['n']
     assert dict(totals)==counts
 for r in REGIMES:
  for h in H:
   rr=[e['test'][f'{m}/{r}/{h}'] for m in methods]
   assert len({(a['n'],a['rmse_v'],a['mae_v'],a['bias_v']) for a in rr})==1
def avg(rows):
 valid=[r for r in rows if r['n']]
 if not valid:return {'n':0,'averaged':0}
 return {'n_mean':statistics.mean(r['n'] for r in valid),'averaged':len(valid)}|{f:statistics.mean(r[f] for r in valid) for f in FIELDS}
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--source',required=True);p.add_argument('--splits',required=True);p.add_argument('--out',required=True);a=p.parse_args()
 root=pathlib.Path(a.root);out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True);src=read(pathlib.Path(a.source));splits=read(pathlib.Path(a.splits));bykey={r['key']:r for r in src['source']}
 rows=[];historical=[];custody=[];fold_groups={}
 for fold in 'YRG':
  path=root/fold;files=verify_manifest(path);r=read(path/'fits'/'results.json')
  assert r['complete'] and r['fold']==fold and r['updates']==600 and r['splits']==splits[fold]
  assert r['source_manifest_sha256']==sha(pathlib.Path(a.source))
  assert len(r['fits'])==15 and {(f['name'],f['seed']) for f in r['fits']}=={(n,s) for n in NAMES for s in SEEDS}
  assert len(r['historical_cog'])==3 and {f['seed'] for f in r['historical_cog']}==set(SEEDS)
  assert r['splits']['train']==src['splits'][fold]['train'] and r['splits']['test']==src['splits'][fold]['test']
  assert set(r['splits']['selection'])|set(r['splits']['calibration'])==set(src['splits'][fold]['dev'])
  groups=[{bykey[k]['pack'] for k in ids} for ids in r['splits'].values()]
  assert all(not groups[i]&groups[j] for i in range(4) for j in range(i)) and len(groups[2])==4
  fold_groups[fold]={k:{'flights':len(v),'pack_labels':len({bykey[x]['pack'] for x in v})} for k,v in r['splits'].items()}
  test=r['splits']['test'];expected_counts={}
  for f in r['fits']:
   assert f['resume_exact']
   ck=path/'fits'/f"{f['seed']}-{f['name']}"
   assert set(f['checkpoint_sha256'])=={'initial.pt','selected.pt','final.pt'}
   for name,expected in f['checkpoint_sha256'].items():assert sha(ck/name)==expected
   e=f['evaluation'];ev(e,METHODS,test,expected_counts)
   rows.append({'fold':fold,'model':f['name'],'seed':f['seed'],'selected':f['selected'],'test':e['test'],'strata':e['strata']})
  for f in r['historical_cog']:
   e=f['evaluation'];ev(e,('original',),test,expected_counts)
   historical.append({'fold':fold,'seed':f['seed'],'test':e['test'],'strata':e['strata']})
  custody.append({'fold':fold,'manifest_files':files,'checkpoint_files':len(list((path/'fits').rglob('*.pt'))),'results_sha256':sha(path/'fits'/'results.json'),'matched_flight_regime_horizon_counts':len(expected_counts)})
  print(fold,'verified',files,'files and all saved metrics',flush=True)
 assert len(rows)==45 and len(historical)==9 and sum(x['checkpoint_files'] for x in custody)==135
 bymodel={}
 for n in NAMES:
  bymodel[n]={}
  for m in METHODS:
   bymodel[n][m]={}
   for reg in REGIMES:
    for h in H:
     key=f'{m}/{reg}/{h}';folds={g:avg([x['test'][key] for x in rows if x['model']==n and x['fold']==g]) for g in 'YRG'}
     bymodel[n][m][f'{reg}/{h}']={'folds':folds,'equal_fold':{f:statistics.mean(z[f] for z in folds.values()) for f in FIELDS}}
 primary={n:{m:bymodel[n][m]['0.15-1/15'] for m in METHODS} for n in NAMES}
 strat={}
 for version,rr,methods in [('v0.1',historical,('original',)),('v0.2',[x for x in rows if x['model']=='cog'],METHODS)]:
  strat[version]={}
  for g in 'YRG':
   strat[version][g]={}
   for m in methods:
    strat[version][g][m]={}
    for reg in REGIMES:
     for h in H:
      for group,bins in STRATA.items():
       for b in bins:
        key=f'{m}/{reg}/{h}/{group}/{b}'
        strat[version][g][m][f'{reg}/{h}/{group}/{b}']=avg([x['strata'][key] for x in rr if x['fold']==g])
 result={'scope':'Independent read-only aggregation of committed sufficient statistics; no checkpoint replay, no refitting, no new data, no independent aircraft guarantee.','custody':custody,'splits':fold_groups,'fit_count':len(rows),'historical_count':len(historical),'primary':primary,'original_primary':{g:avg([x['test']['original/0.15-1/15'] for x in historical if x['fold']==g]) for g in 'YRG'},'qualification':'HOLD'}
 write(out/'summary.json',result);write(out/'all-metrics.json',bymodel);write(out/'cog-strata.json',strat)
 print(json.dumps({'custody':custody,'cog_primary':primary['cog'],'historical':result['original_primary']},indent=2))
if __name__=='__main__':main()
