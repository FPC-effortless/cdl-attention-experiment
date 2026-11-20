"""Publisher TRAIN only; no model, scores, or official DEV / TEST."""
from collections import Counter, defaultdict
from hashlib import sha1, sha256
from pathlib import Path
import json, re, unicodedata
from forge_data.core import canonical
from forge_data.diagnostics import diagnose

ROOT=Path(__file__).resolve().parent
SLOTS=('hotel-area','hotel-pricerange')
EMPTY='<empty>'
HOLD='<hold>'
VALUES=(set('centre east north south west dontcare'.split()),set('cheap moderate expensive dontcare'.split()))

def write(p,obj): p.write_bytes(canonical(obj))
def hashb(b): return sha256(b).hexdigest()
def norm(t): return ' '.join(unicodedata.normalize('NFKC',t).casefold().split())
def feature_key(text):
    return hashb(canonical([norm(t) for t in text]))

def valid_turns(dialogue):
    before=[EMPTY]*2
    context=[];last=-1;rows=[]
    for t in dialogue['turns']:
        idx=int(t['turn_id']);speaker=t['speaker'];text=t['utterance'].lower()
        if idx != last+1 or speaker != ('USER' if idx%2==0 else 'SYSTEM'):
            raise ValueError('unsupported turn order')
        last=idx
        if speaker=='USER':
            states=defaultdict(list)
            for f in t['frames']:
                for key,values in f['state']['slot_values'].items():
                    if key in SLOTS: states[key].append(values)
            after=[]; quality=[]
            for sl,allowed in zip(SLOTS,VALUES):
                found=states[sl]
                if not found: after.append(EMPTY); quality.append('absent')
                elif len(found)!=1 or len(found[0])!=1: after.append(None); quality.append('ambiguous')
                elif found[0][0] not in allowed: after.append(None); quality.append('unknown')
                else: after.append(found[0][0]); quality.append('single')
            y=[None if a is None or b is None else HOLD if a==b else a for a,b in zip(after,before)]
            rows.append(dict(turn=idx,text=[text,*(['']*2+context)[-2:]],delta=y,prior=before,truth=after,quality=quality))
            before=after
        context.append(text)
    return rows

def main():
    dest=ROOT/'data'
    if dest.exists(): raise ValueError('refuse overwrite data freeze')
    publisher=ROOT.parent/'publisher'
    pin=json.loads((ROOT/'PUBLISHER_TRAIN.json').read_text())
    if set(x['name'] for x in pin['files'])!={f'dialogues_{i:03}.json' for i in range(1,18)}: raise ValueError('unexpected publisher inventory')
    paths={}
    byid={}; bysig=defaultdict(list)
    for f in pin['files']:
        p=publisher/'train'/f['name']
        raw=p.read_bytes()
        if len(raw)!=f['bytes'] or sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()!=f['git_blob']:
            raise ValueError('publisher bytes changed')
        paths[f['name']]=hashb(raw)
        for d in json.loads(raw):
            ident=d['dialogue_id']
            if not isinstance(ident,str) or not re.fullmatch(r'[A-Za-z0-9]+\.json',ident) or ident in byid:
                raise ValueError('invalid duplicate original dialogue')
            signature=hashb(canonical([[t['speaker'],norm(t['utterance'])] for t in d['turns']]))
            bysig[signature].append(ident)
            byid[ident]=valid_turns(d)
    partitions=defaultdict(list); counts=defaultdict(Counter); groups=defaultdict(set); group_scope={}; signature_scope={}; features=defaultdict(set)
    for signature,ids in sorted(bysig.items()):
        group=min(ids)
        bucket=int.from_bytes(sha256(('csm-cog062-v1:'+group).encode()).digest()[:8],'big')%100
        part='A' if bucket<60 else 'B' if bucket<80 else 'E'
        signature_scope[signature]=part;group_scope[group]=part
        for ident in sorted(ids):
            counts[part]['dialogues']+=1
            if byid[ident]: groups[part].add(group)
            for r in byid[ident]:
                r.update(dialogue=ident,group=group)
                partitions[part].append(r)
                counts[part]['turns']+=1; features[feature_key(r['text'])].add(part)
                for j,y in enumerate(r['delta']):
                    counts[part][f'{j}:quality:{r["quality"][j]}']+=1
                    if y is None: counts[part][f'{j}:ineligible']+=1
                    elif y==HOLD: counts[part][f'{j}:unchanged']+=1
                    else:
                        kind='removal' if y==EMPTY else 'first_set' if r['prior'][j]==EMPTY else 'overwrite'
                        counts[part][f'{j}:changed']+=1;counts[part][f'{j}:{kind}']+=1
                        counts[part][f'{j}:class:{y}']+=1
    dest.mkdir()
    view=[]
    for p,rr in sorted(partitions.items()):
        write(dest/(p+'.json'),rr)
        for r in rr:
            view.append(dict(id=f"{r['dialogue']}:{r['turn']}",split='validation' if p=='E' else 'train',features={'text_sha256':feature_key(r['text'])},groups={'dialogue_component':r['group']},label=r['delta']))
    write(dest/'forge_view.json',dict(schema='forge.diagnostic-input.v1',feature_fields=['text_sha256'],group_fields=['dialogue_component'],rows=view))
    fv=dest/'forge_view.json'; fd=diagnose(fv,sha256=hashb(fv.read_bytes()),max_rows=100000,max_findings=0)
    write(dest/'forge_receipt.json',fd)
    def qualified_groups(part,slot=None):
        return len({r['group'] for r in partitions[part] if any(y is not None and y!=HOLD for j,y in enumerate(r['delta']) if slot is None or j==slot)})
    scalar={'splits':len(partitions),'dialogue_group_overlap':0,'publisher_train_shards':len(paths),'raw_turns':sum(a['turns'] for a in counts.values()),'primary_changes':sum(counts['E'][f'{j}:changed'] for j in range(2)),'primary_change_groups':qualified_groups('E'),'area_changes':counts['E']['0:changed'],'price_changes':counts['E']['1:changed'],'area_change_groups':qualified_groups('E',0),'price_change_groups':qualified_groups('E',1),'feature_width':3*8192+2+2*8,'meta_max_iter':200,'ordered_arms':2,'base_max_iter':250,'fit_calls':6,'same_meta_rows':1}
    write(dest/'preflight.json',scalar)
    write(dest/'data_audit.json',dict(scope='publisher TRAIN only; normalized feature hashes are diagnostic, not causal decontamination',publisher_train_sha256=paths,publisher_metadata_sha256={n:hashb((publisher/n).read_bytes()) for n in ('README.md','LICENSE','schema.json')},input_sha256={p.name:hashb(p.read_bytes()) for p in sorted(dest.iterdir())},all_counts=counts,groups={k:len(v) for k,v in groups.items()},repeated_dialogue_components=sum(len(z)>1 for z in bysig.values()),cross_partition_feature_hashes=sum(len(z)>1 for z in features.values()),test_reads=0,dev_reads=0,models_fit=0))
    print(json.dumps(scalar,sort_keys=True))
if __name__=='__main__': main()
