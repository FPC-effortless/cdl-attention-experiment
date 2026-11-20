"""Post-hoc saved-artifact adapter; no models or original gates changed."""
import pathlib,json,hashlib,sys,gzip,copy
R=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(R/'veritas539/integrations/trainlab/src'))
from trainlab.competence_panels import freeze_panel,score_panel
from trainlab.prediction_artifacts import make_request
sha=lambda b:hashlib.sha256(b).hexdigest()
canon=lambda o:json.dumps(o,sort_keys=True,separators=(',',':')).encode()
O=R/'reports';O.mkdir(exist_ok=True)
summary=[];mutations=0
for arm in ['parallel','gru']:
 for axisfile in sorted((R/'admitted/seed11/data').glob('dev_*.jsonl')):
  axis=axisfile.stem;records=[json.loads(l) for l in axisfile.read_text().splitlines()];allseeds=[];pins={}
  for seed in [11,23,37]:
   stage=next((R/f'seed{seed}/runs').glob('*/output/stage'));p=stage/f'{arm}_{axis}_primary.json';x=json.loads(p.read_text());allseeds.append((seed,x));pins[str(seed)]={'prediction':sha(p.read_bytes()),'checkpoint':sha((stage/arm/'checkpoint.pt').read_bytes()),'data':sha((R/f'admitted/seed{seed}/data/{axis}.jsonl').read_bytes()),'inputs':sha((R/f'admitted/seed{seed}/INPUTS.json').read_bytes())}
  modes=['queries','sessions_with_guard','paired_first']+(['six_query_regions'] if axis=='dev_independent_regions' else [])
  for mode in modes:
   cases=[];obs=[];inputs=[]
   def add(seed,r,queryid,mapid,expected,actual,eventindex):
    rid=f'{seed}:{r["id"]}:{queryid}';cases.append({'id':rid,'seed':seed,'map_id':mapid,'group':r['pair_id'],'template':axis,'payload':'fixed3byte-or-absent','query_id':str(queryid),'expected':expected});obs.append({'id':rid,'status':'observed','value':actual});inputs.append({'id':rid,'input':{'source_sha256':pins[str(seed)]['prediction'],'dataset_sha256':pins[str(seed)]['data'],'session':r['id'],'event_index':eventindex,'reset':'session start'}})
   for seed,x in allseeds:
    assert len(x['rows'])==len(records)
    for r,p in zip(records,x['rows']):
     assert r['id']==p['id'];first=True
     for i,e in enumerate(r['events']):
      if e['gold']['op_name']!='QUERY':continue
      target=e['gold']['answer'];target=None if target is None else int.from_bytes(target.encode('ascii'),'big')
      expected={'query':True,'valid':True,'value':target};actual={'query':p['actions'][i]['op']=='QUERY','valid':p['errors'][i] is None,'value':p['answers'][i]}
      if mode=='paired_first' and not first:continue
      mapid=r['pair_id'] if mode=='paired_first' else r['id']
      qid=str(r['variant']) if mode=='paired_first' else str(i)
      if mode=='six_query_regions':mapid+=':region'+str(e['gold']['region'])
      add(seed,r,qid,mapid,expected,actual,i);first=False
     if mode=='sessions_with_guard':add(seed,r,'no_unsolicited',r['id'],True,not any(e['gold']['op_name']!='QUERY' and a['op']=='QUERY' for e,a in zip(r['events'],p['actions'])),None)
   ident={k:sha(canon(v)) for k,v in {'model':{s:p['checkpoint'] for s,p in pins.items()},'dataset':{s:p['data'] for s,p in pins.items()},'code':{'adapter':sha(pathlib.Path(__file__).read_bytes()),'veritas':'db9075ec3fa4c4df1a5a02ea993b0fb9ef548b50'},'configuration':{'pins':pins,'mode':mode,'posthoc':True}}.items()}
   request=make_request(ident,inputs,task={'name':'022 posthoc '+mode,'output_schema':'finite-json-v1'},state={'mode':'sequential','initial_state_sha256':sha(b'empty per-session exact store'),'persists':'within original session, offline ordered execution','query_mutates':False,'reset':'reset before each session; no new execution'})
   envelope=freeze_panel(request,cases,required_seeds=[11,23,37],required_templates=[axis],query_mode='continuous',thresholds={'query_accuracy':.95,'map_accuracy':.95});result=score_panel(envelope['panel'],obs,panel_sha256=envelope['panel_sha256'])
   for cell,(seed,x) in zip(result['report']['cells'],allseeds):
    assert cell['seed']==seed
    if mode=='queries':assert cell['correct']==x['scores']['answers_correct'] and cell['queries']==x['scores']['queries']
    if mode=='sessions_with_guard':assert cell['correct_maps']==x['scores']['complete_sessions'] and cell['maps']==x['scores']['sessions']
    if mode=='paired_first':assert cell['correct_maps']==x['scores']['paired_first_correct'] and cell['maps']==x['scores']['pairs']
   bad=copy.deepcopy(envelope['panel']);bad['cases'][0]['expected']='tampered'
   try:score_panel(bad,obs,panel_sha256=envelope['panel_sha256'])
   except Exception:mutations+=1
   else:raise AssertionError('changed expected label accepted')
   name=f'{arm}_{axis}_{mode}';gzip.open(O/(name+'.json.gz'),'wb').write(canon({'pins':pins,'panel':envelope,'observations':obs,'result':result}));summary.append({'arm':arm,'axis':axis,'mode':mode,'panel_sha256':envelope['panel_sha256'],'overall':result['report']['overall'],'cells':result['report']['cells']})
(R/'SUMMARY.json').write_text(json.dumps({'scope':'posthoc exploratory adapter; original gates unchanged','rows':summary,'expected_label_mutation_rejections':mutations},indent=2));print('PASS',len(summary),'panels; mutations',mutations)
