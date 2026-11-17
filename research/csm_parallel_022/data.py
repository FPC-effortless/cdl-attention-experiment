"""022 constructed text sessions. Labels/gold are offline-only, never inference inputs.
Explicit sentence framing is supplied. Semantic roles must be learned from text.
"""
import argparse,hashlib,importlib.util,json,random,re,string,sys
from pathlib import Path
OPS=['NOOP','SET','COPY','LINK','DELETE','QUERY','IFSET']
FIELDS=['dst','src','value','expected']
NAMES={'train':['Ada','Bram','Cleo','Dara','Emil','Faye','Gus','Hana'],'dev':['Ivo','Juno','Kian','Lumi','Neri','Orla','Pia','Quin']}
TEMPLATES={
 'train':{
 'SET':['Set {dst} to {value}.','Put {value} in {dst}.','Assign code {value} to {dst}.'],
 'COPY':['Copy {src} into {dst}.','Give {dst} a snapshot of {src}.','Save a copy of {src} for {dst}.'],
 'LINK':['Link {dst} to {src}.','Let {dst} follow {src}.','Make {dst} refer to {src}.'],
 'DELETE':['Delete {dst}.','Clear {dst}.','Remove the code for {dst}.'],
 'QUERY':['Query {dst}?','What code does {dst} hold?','Read {dst}.'],
 'IFSET':['If {src} holds {expected}, set {dst} to {value}.','Set {dst} to {value} only if {src} is {expected}.','When the code of {src} is {expected}, assign {value} to {dst}.'],
 'NOOP':['The clerk checks the calendar.','Nothing changes during this pause.','The team discusses the weather.']},
 'dev':{
 'SET':['Replace the value of {dst} with {value}.','For {dst}, the new code is {value}.'],
 'COPY':['Retain a snapshot from {src} under {dst}.','Let {dst} preserve the current value of {src}.'],
 'LINK':['Have {dst} track the live value of {src}.','The value at {dst} should mirror {src}.'],
 'DELETE':['Erase the entry for {dst}.','Make {dst} absent.'],
 'QUERY':['Report the current value of {dst}.','Which code belongs to {dst}?'],
 'IFSET':['Provided {src} equals {expected}, write {value} into {dst}.','Give {dst} code {value} when {src} holds {expected}.'],
 'NOOP':['An unrelated notice describes the weather.','No records change while everyone waits.']}}
DEFAULT_REFERENCE=Path(__file__).resolve().parents[1]/'csm-conditional-019-public/research/csm_conditional_019/reference_executor.py'

def load_reference(path=DEFAULT_REFERENCE):
    path=Path(path);name='_typed019_'+hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module.Memory

def tokenize(text):
    matches=list(re.finditer(r'[A-Za-z0-9_/]+|[^\s]',text))
    return {'tokens':[m.group() for m in matches],'offsets':[[m.start(),m.end()] for m in matches]}

def render(op,args,rng,wording='train'):
    template=rng.choice(TEMPLATES[wording][op]);parts=[];spans={f:[-1,-1] for f in FIELDS}
    for literal,field,fmt,conversion in string.Formatter().parse(template):
        parts.append(literal)
        if field is not None:
            start=sum(map(len,parts));parts.append(args[field]);spans[field]=[start,sum(map(len,parts))]
    text=''.join(parts);text.encode('ascii')
    if len(text)>256:raise ValueError('sentence exceeds parser capacity')
    tokens=tokenize(text);token_spans={}
    for field,span in spans.items():
        token_spans[field]=[-1,-1] if span==[-1,-1] else [tokens['offsets'].index(span),tokens['offsets'].index(span)+1]
    return {'text':text,'labels':{'op':OPS.index(op),'spans':spans,'token_spans':token_spans},'tokenization':tokens}

def external_events(episode):
    """The sole intended evaluation input adapter: no hidden labels/roles/state."""
    return [event['text'] for event in episode['events']]

def payload_int(raw):
    if not isinstance(raw,str) or len(raw)!=3 or not re.fullmatch('[A-Z0-9]{3}',raw):raise ValueError('three ASCII byte payload')
    return int.from_bytes(raw.encode('ascii'),'big')
def payload_text(value):return None if value is None else value.to_bytes(3,'big').decode('ascii')

def episode_pair(axis,index,wording='train',names='train',composition=False,delay=0,cross_bank=False,reference=DEFAULT_REFERENCE):
    rng=random.Random(f'CSM-PARALLEL-022:{axis}:{index}');base=rng.sample(NAMES[names],4)
    if cross_bank:
        a,b,c,d=['west/'+base[0],'east/'+base[0],'west/'+base[1],'east/'+base[1]]
    else:a,b,c,d=['west/'+x for x in base]
    values=[]
    while len(values)<4:
        value=''.join(rng.choices('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',k=3))
        if value not in values:values.append(value)
    records=[]
    for variant in [0,1]:
        Memory=load_reference(reference);state=Memory(4);identities={};events=[];qindex=0
        phrasing=random.Random(f'CSM-PARALLEL-022-words:{axis}:{index}')
        def emit(op,dst=None,src=None,value=None,expected=None,query_kind=None):
            nonlocal qindex
            args={k:v for k,v in {'dst':dst,'src':src,'value':value,'expected':expected}.items() if v is not None}
            event=render(op,args,phrasing,wording)
            for name in sorted(set(x for x in (dst,src) if x is not None),key=event['text'].index):
                if name not in identities:identities[name]=len(identities)
            i=identities[dst] if dst is not None else None;j=identities[src] if src is not None else None
            answer=None;guard=None
            if op=='IFSET':
                guard=state.read(j)==payload_int(expected)
                if guard:state.apply('SET',i,v=payload_int(value))
            elif op!='NOOP':answer=state.apply(op,i,j=j,v=payload_int(value) if value else None)
            event['gold']={'op_name':op,'dst_name':dst,'src_name':src,'payload':value,'expected':expected,'answer':payload_text(answer) if op=='QUERY' else None,'guard':guard,'state_after':[{'kind':cell.kind,'payload':cell.payload} for cell in state.cells]}
            if op=='QUERY':event['gold'].update(query_index=qindex,query_kind=query_kind);qindex+=1
            events.append(event)
        emit('SET',a,value=values[variant]);emit('SET',d,value=values[2]);emit('COPY',b,a);emit('LINK',c,a)
        if composition:emit('LINK',c,b) # held additional dependency, not seen TRAIN
        emit('SET',a,value=values[3])
        suffix_event=len(events)
        for _ in range(delay):emit('NOOP')
        emit('QUERY',b,query_kind='snapshot');emit('QUERY',c,query_kind='live_or_chain')
        emit('IFSET',d,b,value=values[3],expected=values[0]);emit('QUERY',d,query_kind='guard_result')
        emit('DELETE',a);emit('QUERY',c,query_kind='after_delete')
        emit('SET',a,value=values[2]);emit('QUERY',c,query_kind='after_recreate');emit('QUERY',b,query_kind='snapshot_preserved')
        records.append({'id':f'{axis}-{index:05d}-v{variant}','pair_id':f'{axis}-{index:05d}','variant':variant,'axis':axis,'events':events,'text':'\n'.join(e['text'] for e in events),'suffix_event':suffix_event,'wording':wording,'names':names,'composition':composition,'delay_events':delay,'cross_bank':cross_bank})
    return records

def regions_pair(axis,index,regions=4,reference=DEFAULT_REFERENCE):
    """Interleave independent bank programs; regions are not independent subjects."""
    if regions!=4:raise ValueError('prospective four-region axis')
    components=[episode_pair(f'{axis}-region{j}',index,delay=4,reference=reference) for j in range(regions)]
    records=[]
    for variant in [0,1]:
        Memory=load_reference(reference);state=Memory(regions*4);identities={};events=[];qindex=0
        phrase=random.Random(f'022regions:{axis}:{index}')
        for step in range(len(components[0][variant]['events'])):
            for region in range(regions):
                source=components[region][variant]['events'][step];gold=source['gold'];op=gold['op_name']
                dst=None if gold['dst_name'] is None else gold['dst_name'].replace('west/',f'bank{region}/')
                src=None if gold['src_name'] is None else gold['src_name'].replace('west/',f'bank{region}/')
                args={k:v for k,v in {'dst':dst,'src':src,'value':gold['payload'],'expected':gold['expected']}.items() if v is not None}
                event=render(op,args,phrase)
                for name in sorted(set(x for x in (dst,src) if x is not None),key=event['text'].index):
                    if name not in identities:identities[name]=len(identities)
                d=identities.get(dst);s=identities.get(src);answer=None;guard=None
                if op=='IFSET':
                    guard=state.read(s)==payload_int(gold['expected'])
                    if guard:state.apply('SET',d,v=payload_int(gold['payload']))
                elif op!='NOOP':answer=state.apply(op,d,j=s,v=payload_int(gold['payload']) if gold['payload'] else None)
                event['gold']={'op_name':op,'dst_name':dst,'src_name':src,'payload':gold['payload'],'expected':gold['expected'],'answer':payload_text(answer) if op=='QUERY' else None,'guard':guard,'region':region,'state_after':[{'kind':c.kind,'payload':c.payload} for c in state.cells]}
                if op=='QUERY':event['gold'].update(query_index=qindex,query_kind=gold['query_kind']);qindex+=1
                events.append(event)
        records.append({'id':f'{axis}-{index:05d}-v{variant}','pair_id':f'{axis}-{index:05d}','variant':variant,'axis':axis,'events':events,'text':'\n'.join(e['text'] for e in events),'suffix_event':5*regions,'wording':'train','names':'train','composition':False,'delay_events':4*regions,'cross_bank':False,'independent_regions':regions})
    return records

def validate_roster(records):
    if len({r['id'] for r in records})!=len(records):raise ValueError('duplicate session ID')
    pairs={}
    for r in records:
        if type(r['variant']) is not int or r['variant'] not in (0,1):raise ValueError('variant')
        pairs.setdefault(r['pair_id'],[]).append(r)
    for pair in pairs.values():
        if len(pair)!=2 or {r['variant'] for r in pair}!={0,1}:raise ValueError('global paired roster')
        a,b=sorted(pair,key=lambda r:r['variant'])
        if external_events(a)[a['suffix_event']:]!=external_events(b)[b['suffix_event']:]:raise ValueError('counterfactual suffix mismatch')
        qa=[e['gold'] for e in a['events'] if e['gold']['op_name']=='QUERY'];qb=[e['gold'] for e in b['events'] if e['gold']['op_name']=='QUERY']
        count=6*a.get('independent_regions',1)
        if len(qa)!=count or len(qb)!=count:raise ValueError('query count')
        for x,y in zip(qa,qb):
            if x['query_kind'] in ('snapshot','guard_result') and x['answer']==y['answer']:raise ValueError('paired answer/guard contrast')
    return {'sessions':len(records),'independent_pairs':len(pairs),'queries':sum(sum(e['gold']['op_name']=='QUERY' for e in r['events']) for r in records)}

def build(output,train_pairs=1024,dev_pairs=64,reference=DEFAULT_REFERENCE):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);manifest=[]
    configs=[('train',train_pairs,'train','train',False,[0,4],False),('dev_familiar',dev_pairs,'train','train',False,[0,4],False),('dev_wording',dev_pairs,'dev','train',False,[0,4],False),('dev_names',dev_pairs,'train','dev',False,[0,4],False),('dev_composition',dev_pairs,'train','train',True,[0,4],False),('dev_delay',dev_pairs,'train','train',False,[16,32],False),('dev_cross_bank',dev_pairs,'train','dev',False,[0,4],True)]
    seen=set()
    for axis,n,w,names,comp,delays,banks in configs:
        rows=[r for i in range(n) for r in episode_pair(axis,i,w,names,comp,delays[i%len(delays)],banks,reference)]
        for r in rows:
            digest=hashlib.sha256(r['text'].encode()).hexdigest()
            if digest in seen:raise ValueError('duplicate complete text across splits')
            seen.add(digest)
        path=output/(axis+'.jsonl');path.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows));manifest.append({'axis':axis,**validate_roster(rows),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'max_sentence_bytes':max(len(e['text']) for r in rows for e in r['events'])})
    rows=[r for i in range(dev_pairs) for r in regions_pair('dev_independent_regions',i,reference=reference)]
    path=output/'dev_independent_regions.jsonl';path.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows));manifest.append({'axis':'dev_independent_regions',**validate_roster(rows),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'max_sentence_bytes':max(len(e['text']) for r in rows for e in r['events'])})
    result={'version':'CSM-PARALLEL-022-prefit-v1','reference_sha256':hashlib.sha256(Path(reference).read_bytes()).hexdigest(),'panels':manifest,'scope':'constructed grammar, supplied sentence boundaries, learned roles required; labels/gold excluded by external_events; no protected TEST'}
    (output/'MANIFEST.json').write_text(json.dumps(result,indent=2));return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--reference',default=str(DEFAULT_REFERENCE));a=p.parse_args();print(json.dumps(build(a.output,reference=a.reference),indent=2))
